//! int8 scalar quantization + an integer distance kernel.
//!
//! WHY
//! ---
//! At 50k x 256d the vectors are 51 MB — already past L2, so every distance
//! call is a stream from DRAM and the AVX-512 float kernel is not compute-bound
//! at all, it is waiting on memory. Shrinking the payload 4x is therefore not a
//! memory optimisation that costs speed, it is a *speed* optimisation that also
//! happens to save memory. That inversion is the whole reason production vector
//! DBs quantize by default.
//!
//! HOW
//! ---
//! Symmetric scalar quantization against one dataset-wide scale:
//!     q_i = round(v_i * 127 / absmax)          in [-127, 127]
//!     ||a-b||^2  ~=  (absmax/127)^2 * sum_i (qa_i - qb_i)^2
//! One global scale (not per-vector) is what keeps the comparison exact in
//! integer space: with per-vector scales you cannot subtract the codes directly
//! and you are back to floats.
//!
//! THE KERNEL
//! ----------
//! `_mm512_dpwssd_epi32` (AVX512-VNNI) is a fused i16 multiply-add: it takes two
//! 32-lane i16 vectors, multiplies pairwise, and adds *pairs of products* into
//! 16 i32 accumulator lanes — one instruction where the float path needs a
//! multiply and an add. We feed it `d` twice to get sum of squares.
//!
//! Widths: 32 i8 per iteration vs 16 f32. Same register, double the elements,
//! quarter the bytes moved.
//!
//! Overflow: |d| <= 254, d^2 <= 64516, two per dpwssd => 129032 per lane per
//! instruction. dim/32 = 8 iterations => 1.03e6, i32 holds 2.1e9. Never wraps.

/// A quantized corpus: codes + the single scale that maps them back.
pub struct Q8 {
    pub codes: Vec<i8>,
    pub dim: usize,
    pub scale: f32, // absmax / 127 — multiply a squared int distance by scale^2
}

impl Q8 {
    pub fn from_f32(data: &[f32], dim: usize) -> Self {
        let absmax = data.iter().fold(0.0f32, |m, &x| m.max(x.abs())).max(1e-12);
        let inv = 127.0 / absmax;
        let codes = data
            .iter()
            .map(|&x| (x * inv).round().clamp(-127.0, 127.0) as i8)
            .collect();
        Q8 { codes, dim, scale: absmax / 127.0 }
    }

    /// Quantize a single query with the corpus scale. Using the corpus scale
    /// (not the query's own) is what makes the codes directly comparable.
    pub fn encode(&self, v: &[f32]) -> Vec<i8> {
        let inv = 1.0 / self.scale;
        v.iter().map(|&x| (x * inv).round().clamp(-127.0, 127.0) as i8).collect()
    }

    #[inline]
    pub fn at(&self, id: u32) -> &[i8] {
        let s = id as usize * self.dim;
        &self.codes[s..s + self.dim]
    }

    /// Squared L2 back in float units, for comparing against the f32 index.
    #[inline]
    pub fn to_f32(&self, raw: i32) -> f32 {
        raw as f32 * self.scale * self.scale
    }

    pub fn bytes(&self) -> usize { self.codes.len() }
}

/// Scalar reference. i32 accumulation, so it is bit-exact against the SIMD path
/// — which is the point of quantizing: the test can assert equality, not a
/// tolerance. Float kernels can only ever be compared with an epsilon.
pub fn l2sq_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    let mut s = 0i32;
    for i in 0..a.len() {
        let d = a[i] as i32 - b[i] as i32;
        s += d * d;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn l2sq_i8_vnni(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // Two accumulators: dpwssd has ~5 cycle latency and ~1/cycle throughput, so
    // a single accumulator would sit latency-bound at 20% of peak.
    let mut acc0 = _mm512_setzero_si512();
    let mut acc1 = _mm512_setzero_si512();
    let mut i = 0;
    while i + 64 <= n {
        // sign-extend 32 x i8 -> 32 x i16. The subtract MUST happen at i16:
        // at i8 the difference overflows (127 - (-127) = 254).
        let a0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(pa.add(i) as *const __m256i));
        let b0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(pb.add(i) as *const __m256i));
        let d0 = _mm512_sub_epi16(a0, b0);
        acc0 = _mm512_dpwssd_epi32(acc0, d0, d0);

        let a1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(pa.add(i + 32) as *const __m256i));
        let b1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(pb.add(i + 32) as *const __m256i));
        let d1 = _mm512_sub_epi16(a1, b1);
        acc1 = _mm512_dpwssd_epi32(acc1, d1, d1);
        i += 64;
    }
    while i + 32 <= n {
        let a0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(pa.add(i) as *const __m256i));
        let b0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256(pb.add(i) as *const __m256i));
        let d0 = _mm512_sub_epi16(a0, b0);
        acc0 = _mm512_dpwssd_epi32(acc0, d0, d0);
        i += 32;
    }
    let mut s = _mm512_reduce_add_epi32(_mm512_add_epi32(acc0, acc1));
    while i < n {
        let d = *pa.add(i) as i32 - *pb.add(i) as i32;
        s += d * d;
        i += 1;
    }
    s
}

pub type DistI8 = fn(&[i8], &[i8]) -> i32;

pub fn best_l2sq_i8() -> (DistI8, &'static str) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512f")
        {
            return (|a, b| unsafe { l2sq_i8_vnni(a, b) }, "avx512vnni");
        }
    }
    (l2sq_i8_scalar, "scalar-i8")
}

/// Brute-force top-k over the quantized corpus. Used to measure how much recall
/// quantization actually costs, independently of the graph.
pub fn brute_topk(q8: &Q8, qcode: &[i8], k: usize, f: DistI8) -> Vec<(u32, i32)> {
    let n = q8.codes.len() / q8.dim;
    let mut all: Vec<(u32, i32)> = (0..n as u32).map(|i| (i, f(qcode, q8.at(i)))).collect();
    all.sort_unstable_by_key(|x| x.1);
    all.truncate(k);
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Integer kernels admit exact tests. Any disagreement is a real bug, not
    /// float reassociation.
    #[test]
    fn vnni_matches_scalar_exactly() {
        let (f, name) = best_l2sq_i8();
        for dim in [7usize, 32, 64, 96, 255, 256, 768] {
            let a: Vec<i8> = (0..dim).map(|i| ((i * 37 % 255) as i32 - 127) as i8).collect();
            let b: Vec<i8> = (0..dim).map(|i| ((i * 91 % 255) as i32 - 127) as i8).collect();
            assert_eq!(f(&a, &b), l2sq_i8_scalar(&a, &b), "{name} dim={dim}");
        }
    }

    /// Worst case for overflow: every lane at maximum separation.
    #[test]
    fn no_overflow_at_max_separation() {
        let (f, _) = best_l2sq_i8();
        let dim = 4096;
        let a = vec![127i8; dim];
        let b = vec![-127i8; dim];
        assert_eq!(f(&a, &b), 254i32 * 254 * dim as i32);
    }
}
