//! Synthetic corpora. Kept in the library, not the bench, because "what data
//! did you measure on" is the single most load-bearing fact in any ANN number.

pub struct R(pub u64);
impl R {
    pub fn f(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.0 = x;
        ((x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / ((1u32 << 24) as f32)
    }
    pub fn gauss(&mut self) -> f32 {
        let u1 = self.f().max(1e-7);
        let u2 = self.f();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Low intrinsic rank in a high ambient dimension — the shape real embeddings
/// actually have. `rank` is the knob that controls ANN difficulty; `dim` is the
/// number you put in the index config and it controls almost nothing.
pub fn lowrank(n: usize, dim: usize, rank: usize, seed: u64) -> Vec<f32> {
    let mut r = R(seed | 1);
    let proj: Vec<f32> = (0..rank * dim).map(|_| r.gauss()).collect();
    let mut out = vec![0.0f32; n * dim];
    for i in 0..n {
        let lat: Vec<f32> = (0..rank).map(|_| r.gauss()).collect();
        for j in 0..dim {
            let mut acc = 0.0;
            for t in 0..rank { acc += lat[t] * proj[t * dim + j]; }
            out[i * dim + j] = acc / (rank as f32).sqrt();
        }
    }
    out
}
