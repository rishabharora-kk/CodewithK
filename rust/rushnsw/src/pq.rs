//! Product quantization: 32x compression, and a distance kernel that does no
//! arithmetic on the data at all.
//!
//! THE IDEA
//! --------
//! Scalar quantization (`q8.rs`) shrinks each *number* from 4 bytes to 1. That
//! caps out at 4x. Product quantization shrinks each *subvector* instead: chop
//! a 256-d vector into m=32 chunks of 8 dims, run k-means with 256 centroids
//! inside each chunk, and store only which centroid each chunk landed on. One
//! byte per chunk. A 1024-byte vector becomes 32 bytes.
//!
//! The reconstruction is a product of m independent codebooks, hence the name:
//! 256^32 representable vectors from 32*256 stored centroids.
//!
//! THE KERNEL IS A DIFFERENT SHAPE
//! -------------------------------
//! This is the part worth internalising. For `l2sq` and the int8 kernel, the
//! query and the stored vector both stream through the ALU. For PQ they don't.
//! Because every stored vector is built from the same 32*256 centroids, you
//! precompute — ONCE per query — the distance from each query chunk to all 256
//! centroids of that chunk. That is a 32x256 table, 8192 floats.
//!
//! After that, the distance to any stored vector is 32 table lookups and 32
//! adds. No multiplies. No touching the original data. The vector's 32 bytes
//! are *indices*, not values.
//!
//! So the kernel stops being FMA-bound and becomes gather-bound, and the
//! interesting number stops being GFLOP/s and becomes whether the table stays
//! in L1. This is called ADC — asymmetric distance computation — asymmetric
//! because the query stays in full precision and only the database is
//! quantized. That asymmetry is free accuracy: there is no reason to degrade
//! the one vector you have in full precision.

use rayon::prelude::*;

pub struct Pq {
    pub m: usize,     // subquantizers
    pub dsub: usize,  // dims per subquantizer
    pub ksub: usize,  // centroids per subquantizer (256 => codes are u8)
    pub dim: usize,
    /// m * ksub * dsub, laid out so one subquantizer's codebook is contiguous.
    pub centroids: Vec<f32>,
    /// n * m, one byte per subquantizer. THIS is the index payload.
    pub codes: Vec<u8>,
    pub n: usize,
}

struct R(u64);
impl R {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
}

#[inline]
fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0.0;
    for i in 0..a.len() { let d = a[i] - b[i]; s += d * d; }
    s
}

/// Lloyd's algorithm inside one subspace. dsub is small (8), so this is
/// deliberately plain code — the compiler auto-vectorises an 8-wide inner loop
/// better than hand-written intrinsics would at this size, and the honest place
/// to spend intrinsics is the query-time kernel, not the once-per-index train.
fn kmeans_subspace(train: &[f32], nt: usize, dsub: usize, ksub: usize, iters: usize, seed: u64)
    -> Vec<f32>
{
    let mut r = R(seed | 1);
    let mut cent = vec![0.0f32; ksub * dsub];
    // init: distinct random training points
    for c in 0..ksub {
        let i = (r.next() as usize) % nt;
        cent[c * dsub..(c + 1) * dsub].copy_from_slice(&train[i * dsub..(i + 1) * dsub]);
    }

    let mut assign = vec![0u32; nt];
    let mut sums = vec![0.0f32; ksub * dsub];
    let mut counts = vec![0u32; ksub];

    for _ in 0..iters {
        for i in 0..nt {
            let v = &train[i * dsub..(i + 1) * dsub];
            let mut best = f32::INFINITY;
            let mut bi = 0u32;
            for c in 0..ksub {
                let d = l2sq(v, &cent[c * dsub..(c + 1) * dsub]);
                if d < best { best = d; bi = c as u32; }
            }
            assign[i] = bi;
        }
        sums.iter_mut().for_each(|x| *x = 0.0);
        counts.iter_mut().for_each(|x| *x = 0);
        for i in 0..nt {
            let c = assign[i] as usize;
            counts[c] += 1;
            let v = &train[i * dsub..(i + 1) * dsub];
            for j in 0..dsub { sums[c * dsub + j] += v[j]; }
        }
        for c in 0..ksub {
            if counts[c] == 0 {
                // An empty cluster is wasted codebook capacity — 1/256th of this
                // subquantizer's resolution doing nothing. Re-seed it onto a
                // random point rather than leaving it stranded.
                let i = (r.next() as usize) % nt;
                cent[c * dsub..(c + 1) * dsub].copy_from_slice(&train[i * dsub..(i + 1) * dsub]);
            } else {
                let inv = 1.0 / counts[c] as f32;
                for j in 0..dsub { cent[c * dsub + j] = sums[c * dsub + j] * inv; }
            }
        }
    }
    cent
}

impl Pq {
    /// Train on (a sample of) the corpus, then encode all of it.
    /// Subspaces are independent by construction, which is why this parallelises
    /// with a bare `par_iter` and no locks at all — the opposite of `par.rs`.
    pub fn train(data: &[f32], dim: usize, m: usize, iters: usize, max_train: usize, seed: u64)
        -> Self
    {
        assert_eq!(dim % m, 0, "dim must divide evenly into m subquantizers");
        let dsub = dim / m;
        let ksub = 256;
        let n = data.len() / dim;
        let nt = n.min(max_train);
        let stride = (n / nt).max(1);

        let centroids: Vec<f32> = (0..m)
            .into_par_iter()
            .flat_map_iter(|j| {
                // gather this subspace's training slice contiguously
                let mut tr = Vec::with_capacity(nt * dsub);
                for t in 0..nt {
                    let i = t * stride;
                    let s = i * dim + j * dsub;
                    tr.extend_from_slice(&data[s..s + dsub]);
                }
                kmeans_subspace(&tr, nt, dsub, ksub, iters, seed ^ (j as u64 + 1))
                    .into_iter()
            })
            .collect();

        let mut pq = Pq { m, dsub, ksub, dim, centroids, codes: Vec::new(), n };
        pq.codes = pq.encode_many(data);
        pq
    }

    pub fn encode_into(&self, v: &[f32], out: &mut [u8]) {
        for j in 0..self.m {
            let sv = &v[j * self.dsub..(j + 1) * self.dsub];
            let base = j * self.ksub * self.dsub;
            let mut best = f32::INFINITY;
            let mut bi = 0u8;
            for c in 0..self.ksub {
                let d = l2sq(sv, &self.centroids[base + c * self.dsub..base + (c + 1) * self.dsub]);
                if d < best { best = d; bi = c as u8; }
            }
            out[j] = bi;
        }
    }

    pub fn encode_many(&self, data: &[f32]) -> Vec<u8> {
        let n = data.len() / self.dim;
        let mut codes = vec![0u8; n * self.m];
        codes
            .par_chunks_mut(self.m)
            .enumerate()
            .for_each(|(i, out)| self.encode_into(&data[i * self.dim..(i + 1) * self.dim], out));
        codes
    }

    #[inline]
    pub fn code(&self, id: u32) -> &[u8] {
        let s = id as usize * self.m;
        &self.codes[s..s + self.m]
    }

    pub fn bytes(&self) -> usize { self.codes.len() }
    pub fn codebook_bytes(&self) -> usize { self.centroids.len() * 4 }

    /// Build the per-query distance table: m * ksub floats. Computed once, then
    /// amortised over every node the search touches. At m=32 this is 8192 f32 =
    /// 32 KB, which is right at the L1D boundary on most cores — the single most
    /// important number for PQ query performance, and the reason production
    /// systems quantize the *table* to u8 as well.
    pub fn lut(&self, q: &[f32]) -> Vec<f32> {
        let mut t = vec![0.0f32; self.m * self.ksub];
        for j in 0..self.m {
            let sq = &q[j * self.dsub..(j + 1) * self.dsub];
            let base = j * self.ksub * self.dsub;
            for c in 0..self.ksub {
                t[j * self.ksub + c] =
                    l2sq(sq, &self.centroids[base + c * self.dsub..base + (c + 1) * self.dsub]);
            }
        }
        t
    }

    pub fn lut_bytes(&self) -> usize { self.m * self.ksub * 4 }
}

/// Scalar ADC: 32 lookups, 32 adds, zero multiplies.
pub fn adc_scalar(lut: &[f32], code: &[u8], ksub: usize) -> f32 {
    let mut s = 0.0f32;
    for j in 0..code.len() {
        s += lut[j * ksub + code[j] as usize];
    }
    s
}

/// AVX-512 ADC via gather.
///
/// `_mm512_i32gather_ps` fetches 16 floats from 16 independent addresses in one
/// instruction. That is the whole kernel: widen 16 code bytes to i32, add the
/// per-subquantizer row offsets, gather, accumulate.
///
/// Gather is not cheap — roughly a load per element internally, ~20 cycle
/// latency — so this does NOT beat a dense SIMD kernel on arithmetic. It wins
/// when the dense kernel would have to stream 1 KB per vector from DRAM and
/// this one reads 32 bytes. That crossover is a property of the dataset size,
/// not of the instruction, which is why the benchmark reports both.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn adc_avx512(lut: &[f32], code: &[u8], ksub: usize) -> f32 {
    use std::arch::x86_64::*;
    debug_assert_eq!(ksub, 256);
    let m = code.len();
    let base = lut.as_ptr();
    // row offsets: subquantizer j's table starts at j*256 floats
    let step = _mm512_setr_epi32(0, 256, 512, 768, 1024, 1280, 1536, 1792,
                                 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840);
    let bump = _mm512_set1_epi32(4096); // 16 subquantizers later
    let mut acc = _mm512_setzero_ps();
    let mut j = 0;
    let mut rowbase = step;
    while j + 16 <= m {
        let c = _mm512_cvtepu8_epi32(_mm_loadu_si128(code.as_ptr().add(j) as *const __m128i));
        let idx = _mm512_add_epi32(rowbase, c);
        acc = _mm512_add_ps(acc, _mm512_i32gather_ps(idx, base, 4));
        rowbase = _mm512_add_epi32(rowbase, bump);
        j += 16;
    }
    let mut s = _mm512_reduce_add_ps(acc);
    while j < m {
        s += *lut.get_unchecked(j * ksub + *code.get_unchecked(j) as usize);
        j += 1;
    }
    s
}

pub type AdcFn = fn(&[f32], &[u8], usize) -> f32;

pub fn best_adc() -> (AdcFn, &'static str) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return (|l, c, k| unsafe { adc_avx512(l, c, k) }, "avx512-gather");
        }
    }
    (adc_scalar, "scalar-adc")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::lowrank;

    #[test]
    fn gather_matches_scalar() {
        let (f, _) = best_adc();
        let m = 32;
        let lut: Vec<f32> = (0..m * 256).map(|i| (i as f32) * 0.001).collect();
        for trial in 0..8u32 {
            let code: Vec<u8> = (0..m).map(|j| ((j as u32 * 37 + trial * 91) % 256) as u8).collect();
            let a = f(&lut, &code, 256);
            let b = adc_scalar(&lut, &code, 256);
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    /// The real correctness property: ADC against the codebook must approximate
    /// true L2. If training silently produced garbage centroids this catches it,
    /// where a kernel-vs-kernel test would not.
    #[test]
    fn adc_approximates_true_distance() {
        let (dim, n) = (64usize, 2000usize);
        let data = lowrank(n, dim, 16, 5);
        let pq = Pq::train(&data, dim, 8, 10, 2000, 42);
        let (f, _) = best_adc();
        let q = &data[7 * dim..8 * dim];
        let lut = pq.lut(q);

        // rank correlation is what actually matters for search, so check that
        // the nearest neighbour by ADC is genuinely near by exact distance.
        let mut exact: Vec<(f32, u32)> = (0..n as u32)
            .map(|i| (l2sq(q, &data[i as usize * dim..(i as usize + 1) * dim]), i)).collect();
        exact.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let top: Vec<u32> = exact[..50].iter().map(|x| x.1).collect();

        let mut approx: Vec<(f32, u32)> = (0..n as u32)
            .map(|i| (f(&lut, pq.code(i), 256), i)).collect();
        approx.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let hits = approx[..10].iter().filter(|x| top.contains(&x.1)).count();
        assert!(hits >= 7, "PQ top-10 kept only {hits}/10 inside exact top-50");
    }

    #[test]
    fn compression_is_what_it_claims() {
        let (dim, n) = (256usize, 500usize);
        let data = lowrank(n, dim, 32, 1);
        let pq = Pq::train(&data, dim, 32, 5, 500, 9);
        assert_eq!(pq.bytes(), n * 32);
        assert_eq!(data.len() * 4 / pq.bytes(), 32);
    }
}
