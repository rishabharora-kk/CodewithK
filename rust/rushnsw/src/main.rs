use rushnsw::dist::{best_l2sq, l2sq_scalar};
use rushnsw::hnsw::Hnsw;
use std::time::Instant;

struct R(u64);
impl R {
    fn f(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.0 = x;
        ((x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / ((1u32 << 24) as f32)
    }
    fn gauss(&mut self) -> f32 {
        // Box-Muller, one sample
        let u1 = self.f().max(1e-7);
        let u2 = self.f();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

/// Clustered data — uniform-random vectors are pathologically easy for ANN
/// (everything is equidistant in high dim). Real embeddings live on clusters.
fn make_data(n: usize, dim: usize, nclust: usize, seed: u64) -> Vec<f32> {
    let mut r = R(seed | 1);
    let centers: Vec<f32> = (0..nclust * dim).map(|_| r.gauss() * 4.0).collect();
    let mut out = Vec::with_capacity(n * dim);
    for i in 0..n {
        let c = (i % nclust) * dim;
        for j in 0..dim {
            out.push(centers[c + j] + r.gauss());
        }
    }
    out
}

/// Real embeddings have low *intrinsic* dimension: a 256-d BERT vector does
/// not fill 256 dims, it lies near a ~10-30 dim manifold. Generate that:
/// sample a latent, project through a fixed random matrix.
fn make_lowrank(n: usize, dim: usize, rank: usize, seed: u64) -> Vec<f32> {
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

fn sweep(tag: &str, data: &[f32], queries: &[f32], dim: usize, n: usize, nq: usize, k: usize, dfn: fn(&[f32],&[f32])->f32) {
    let mut idx = Hnsw::new(dim, 16, 200, 1234);
    let t = Instant::now();
    for i in 0..n { idx.insert(&data[i * dim..(i + 1) * dim]); }
    let build = t.elapsed();
    let mut truth = Vec::with_capacity(nq);
    for qi in 0..nq {
        let q = &queries[qi * dim..(qi + 1) * dim];
        let mut all: Vec<(f32, u32)> = (0..n as u32)
            .map(|i| (dfn(q, &data[i as usize * dim..(i as usize + 1) * dim]), i)).collect();
        all.sort_unstable_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        truth.push(all[..k].iter().map(|x| x.1).collect::<Vec<_>>());
    }
    println!("\n== {tag} == (build {:.1?})", build);
    println!("   ef |  recall | dist/query");
    for ef in [16, 32, 64, 128] {
        idx.dist_calls = 0;
        let mut hit = 0usize;
        for qi in 0..nq {
            for (id, _) in &idx.search(&queries[qi*dim..(qi+1)*dim], k, ef) {
                if truth[qi].contains(id) { hit += 1; }
            }
        }
        println!("  {ef:>3} |  {:.3}  | {:>10.0}", hit as f64/(nq*k) as f64, idx.dist_calls as f64/nq as f64);
    }
}

fn main() {
    let dim = 256;
    let n = 50_000;
    let nq = 500;
    let k = 10;

    let (dfn, kernel) = best_l2sq();
    println!("kernel: {kernel}   n={n} dim={dim}\n");

    // ---- 1. kernel microbenchmark ----
    let a: Vec<f32> = (0..dim).map(|i| i as f32 * 0.01).collect();
    let b: Vec<f32> = (0..dim).map(|i| (dim - i) as f32 * 0.013).collect();
    let iters = 3_000_000u64;
    let t = Instant::now();
    let mut s = 0.0f32;
    for _ in 0..iters { s += l2sq_scalar(&a, &b); }
    let scalar_ns = t.elapsed().as_nanos() as f64 / iters as f64;
    let t = Instant::now();
    let mut s2 = 0.0f32;
    for _ in 0..iters { s2 += dfn(&a, &b); }
    let simd_ns = t.elapsed().as_nanos() as f64 / iters as f64;
    assert!((s - s2).abs() / s.abs() < 1e-3, "kernels disagree");
    println!("== distance kernel ({dim}d) ==");
    println!("  scalar : {scalar_ns:7.1} ns  ({:.1} GFLOP/s)", 3.0 * dim as f64 / scalar_ns);
    println!("  {kernel:<7}: {simd_ns:7.1} ns  ({:.1} GFLOP/s)   speedup {:.2}x\n",
             3.0 * dim as f64 / simd_ns, scalar_ns / simd_ns);

    // ---- 2. build ----
    let data = make_data(n, dim, 200, 42);
    let queries = make_data(nq, dim, 200, 7);

    let mut idx = Hnsw::new(dim, 16, 200, 1234);
    let t = Instant::now();
    for i in 0..n { idx.insert(&data[i * dim..(i + 1) * dim]); }
    let build = t.elapsed();
    println!("== build ==");
    println!("  {n} vectors in {:.2?}  ({:.0}/s), top level {}\n",
             build, n as f64 / build.as_secs_f64(), idx.max_level());

    // ---- 3. ground truth ----
    let t = Instant::now();
    let mut truth = Vec::with_capacity(nq);
    for qi in 0..nq {
        let q = &queries[qi * dim..(qi + 1) * dim];
        let mut all: Vec<(f32, u32)> = (0..n as u32)
            .map(|i| (dfn(q, &data[i as usize * dim..(i as usize + 1) * dim]), i))
            .collect();
        all.sort_unstable_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        truth.push(all[..k].iter().map(|x| x.1).collect::<Vec<_>>());
    }
    let bf = t.elapsed();
    let bf_qps = nq as f64 / bf.as_secs_f64();
    println!("== brute force baseline ==");
    println!("  {:.2} QPS  ({:.2} ms/query, {n} dist/query)\n", bf_qps, 1000.0 / bf_qps);

    print!("== sanity ==\n"); selftest(&mut idx, &data, dim, n);

    // ---- 4. recall / QPS sweep ----
    println!("== HNSW recall@{k} vs ef ==");
    println!("   ef |  recall |     QPS | speedup | dist/query");
    println!("  ----+---------+---------+---------+-----------");
    for ef in [10, 16, 32, 64, 128, 256] {
        idx.dist_calls = 0;
        let t = Instant::now();
        let mut hit = 0usize;
        for qi in 0..nq {
            let q = &queries[qi * dim..(qi + 1) * dim];
            let got = idx.search(q, k, ef);
            for (id, _) in &got {
                if truth[qi].contains(id) { hit += 1; }
            }
        }
        let el = t.elapsed();
        let qps = nq as f64 / el.as_secs_f64();
        println!("  {ef:>3} |  {:.3}  | {qps:>7.0} | {:>6.0}x | {:>10.0}",
                 hit as f64 / (nq * k) as f64,
                 qps / bf_qps,
                 idx.dist_calls as f64 / nq as f64);
    }

    // ---- 5. does intrinsic dimension explain the low recall? ----
    for rank in [8usize, 32, 128] {
        let d2 = make_lowrank(n, dim, rank, 99);
        let q2 = make_lowrank(nq, dim, rank, 99);
        let q2: Vec<f32> = q2.iter().map(|x| x * 1.0).collect();
        sweep(&format!("intrinsic rank {rank} (ambient {dim})"), &d2, &q2, dim, n, 200, k, dfn);
    }
}

// sanity: query with vectors that are IN the index. recall@1 must be ~1.0
// if the graph is correct. Anything less is a wiring bug, not a tuning issue.

fn selftest(idx: &mut Hnsw, data: &[f32], dim: usize, n: usize) {
    let mut ok = 0;
    for i in (0..n).step_by(n / 500) {
        let r = idx.search(&data[i * dim..(i + 1) * dim], 1, 32);
        if r[0].0 == i as u32 { ok += 1; }
    }
    println!("  self-retrieval recall@1 (ef=32): {:.3}", ok as f64 / ((n / (n/500)) as f64).max(1.0));
}

