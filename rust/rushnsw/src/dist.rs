//! Distance kernels. Scalar baseline + AVX2/FMA + AVX-512 with runtime dispatch.
//!
//! The whole point: in ANN search, >90% of wall clock is this function.
//! Everything else in HNSW is bookkeeping around making fewer of these calls
//! and making each one cheaper.

/// Squared L2. We never sqrt — monotonic, and sqrt costs ~15 cycles.
#[inline]
pub fn l2sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2sq_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    // 4 independent accumulators: breaks the FMA dependency chain.
    // One accumulator => latency-bound (~4 cyc/FMA). Four => throughput-bound.
    let mut acc = [unsafe { _mm256_setzero_ps() }; 4];
    let mut i = 0;
    while i + 32 <= n {
        for k in 0..4 {
            let off = i + k * 8;
            let va = unsafe { _mm256_loadu_ps(pa.add(off)) };
            let vb = unsafe { _mm256_loadu_ps(pb.add(off)) };
            let d = unsafe { _mm256_sub_ps(va, vb) };
            acc[k] = unsafe { _mm256_fmadd_ps(d, d, acc[k]) };
        }
        i += 32;
    }
    let mut v = unsafe { _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3])) };
    while i + 8 <= n {
        let va = unsafe { _mm256_loadu_ps(pa.add(i)) };
        let vb = unsafe { _mm256_loadu_ps(pb.add(i)) };
        let d = unsafe { _mm256_sub_ps(va, vb) };
        v = unsafe { _mm256_fmadd_ps(d, d, v) };
        i += 8;
    }
    // horizontal reduce
    let lo = unsafe { _mm256_castps256_ps128(v) };
    let hi = unsafe { _mm256_extractf128_ps(v, 1) };
    let mut s128 = unsafe { _mm_add_ps(lo, hi) };
    s128 = unsafe { _mm_hadd_ps(s128, s128) };
    s128 = unsafe { _mm_hadd_ps(s128, s128) };
    let mut s = unsafe { _mm_cvtss_f32(s128) };
    while i < n {
        let d = a[i] - b[i];
        s += d * d;
        i += 1;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2sq_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc = [unsafe { _mm512_setzero_ps() }; 2];
    let mut i = 0;
    while i + 32 <= n {
        for k in 0..2 {
            let off = i + k * 16;
            let va = unsafe { _mm512_loadu_ps(pa.add(off)) };
            let vb = unsafe { _mm512_loadu_ps(pb.add(off)) };
            let d = unsafe { _mm512_sub_ps(va, vb) };
            acc[k] = unsafe { _mm512_fmadd_ps(d, d, acc[k]) };
        }
        i += 32;
    }
    let mut v = unsafe { _mm512_add_ps(acc[0], acc[1]) };
    while i + 16 <= n {
        let va = unsafe { _mm512_loadu_ps(pa.add(i)) };
        let vb = unsafe { _mm512_loadu_ps(pb.add(i)) };
        let d = unsafe { _mm512_sub_ps(va, vb) };
        v = unsafe { _mm512_fmadd_ps(d, d, v) };
        i += 16;
    }
    let mut s = unsafe { _mm512_reduce_add_ps(v) };
    while i < n {
        let d = a[i] - b[i];
        s += d * d;
        i += 1;
    }
    s
}

/// A function pointer resolved ONCE at index construction, not per call.
/// `is_x86_feature_detected!` is cheap but not free; hoisting it out of the
/// hot loop is the difference between a branch and a dispatch per 10M calls.
pub type DistFn = fn(&[f32], &[f32]) -> f32;

pub fn best_l2sq() -> (DistFn, &'static str) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return (|a, b| unsafe { l2sq_avx512(a, b) }, "avx512f");
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return (|a, b| unsafe { l2sq_avx2(a, b) }, "avx2+fma");
        }
    }
    (l2sq_scalar, "scalar")
}
