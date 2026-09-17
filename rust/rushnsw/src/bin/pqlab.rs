//! Round 3: product quantization. Does a 32x smaller index still find things,
//! and is a gather actually faster than arithmetic?
//!
//! Run at two corpus sizes on purpose. PQ's win is a memory-traffic win, and a
//! memory-traffic win is invisible while the corpus fits in cache.

use rushnsw::dist::best_l2sq;
use rushnsw::hnsw::Hnsw;
use rushnsw::par;
use rushnsw::pq::{adc_scalar, best_adc, Pq};
use rushnsw::q8::{best_l2sq_i8, Q8};
use rushnsw::synth::lowrank;
use std::time::Instant;

const DIM: usize = 256;
const NQ: usize = 300;
const K: usize = 10;
const RANK: usize = 32;

fn recall(got: &[Vec<u32>], truth: &[Vec<u32>]) -> f64 {
    let mut hit = 0;
    for (g, t) in got.iter().zip(truth) {
        for id in g { if t.contains(id) { hit += 1; } }
    }
    hit as f64 / (truth.len() * K) as f64
}

fn run(n: usize) {
    let (df, _) = best_l2sq();
    println!("\n{}", "=".repeat(78));
    println!("n = {n}   dim = {DIM}   intrinsic rank = {RANK}   f32 corpus = {:.1} MB",
             (n * DIM * 4) as f64 / 1e6);
    println!("{}", "=".repeat(78));

    let data = lowrank(n, DIM, RANK, 99);
    let queries = lowrank(NQ, DIM, RANK, 99);

    let truth: Vec<Vec<u32>> = {
        let t = Instant::now();
        let r: Vec<Vec<u32>> = (0..NQ).map(|qi| {
            let q = &queries[qi*DIM..(qi+1)*DIM];
            let mut all: Vec<(f32,u32)> = (0..n as u32)
                .map(|i| (df(q, &data[i as usize*DIM..(i as usize+1)*DIM]), i)).collect();
            all.sort_unstable_by(|a,b| a.0.partial_cmp(&b.0).unwrap());
            all[..K].iter().map(|x| x.1).collect()
        }).collect();
        println!("exact ground truth: {:.1?}", t.elapsed());
        r
    };

    let t = Instant::now();
    let mut idx = par::build(data.clone(), DIM, 16, 200, 1234);
    println!("graph build:        {:.1?}", t.elapsed());

    idx.quantize();
    let t = Instant::now();
    idx.train_pq(32, 15, 25_000, 7);
    println!("pq train (m=32):    {:.1?}\n", t.elapsed());

    // ---- memory ----
    let g = idx.graph_bytes();
    println!("  payload per vector      total (+ graph {:.1} MB)", g as f64/1e6);
    println!("  f32   {:>4} B          {:>7.1} MB", DIM*4, (idx.f32_bytes()+g) as f64/1e6);
    println!("  int8  {:>4} B          {:>7.1} MB   {:.0}x smaller payload", DIM,
             (idx.q8_bytes()+g) as f64/1e6, 4.0);
    println!("  pq32  {:>4} B          {:>7.1} MB   {:.0}x smaller payload   (+{:.0} KB codebook)",
             32, (idx.pq_bytes()+g) as f64/1e6, (DIM*4) as f64/32.0,
             idx.pq_codebook_bytes() as f64/1e3);
    println!("  per-query LUT: {:.0} KB (built once, then {} gathers per node)\n",
             idx.pq_lut_bytes() as f64/1e3, 32);

    // ---- kernels ----
    let q8 = Q8::from_f32(&data, DIM);
    let pq = Pq::train(&data[..(n.min(20_000))*DIM], DIM, 32, 5, 10_000, 3);
    let (di, _) = best_l2sq_i8();
    let (adc, adck) = best_adc();
    let (ca, cb) = (q8.at(0).to_vec(), q8.at(1).to_vec());
    let lut = pq.lut(&data[..DIM]);
    let code = pq.code(1).to_vec();
    let iters = 3_000_000u64;

    let t = Instant::now(); let mut x = 0.0f32;
    for _ in 0..iters { x += df(&data[..DIM], &data[DIM..2*DIM]); }
    let a = t.elapsed().as_nanos() as f64/iters as f64;
    let t = Instant::now(); let mut y = 0i64;
    for _ in 0..iters { y += di(&ca, &cb) as i64; }
    let b = t.elapsed().as_nanos() as f64/iters as f64;
    let t = Instant::now(); let mut z = 0.0f32;
    for _ in 0..iters { z += adc(&lut, &code, 256); }
    let c = t.elapsed().as_nanos() as f64/iters as f64;
    let t = Instant::now(); let mut w = 0.0f32;
    for _ in 0..iters { w += adc_scalar(&lut, &code, 256); }
    let d = t.elapsed().as_nanos() as f64/iters as f64;
    std::hint::black_box((x,y,z,w));
    println!("  kernel, hot cache       ns/call");
    println!("  f32  avx512             {a:6.1}");
    println!("  int8 vnni               {b:6.1}");
    println!("  pq   {adck:<18} {c:6.1}");
    println!("  pq   scalar             {d:6.1}\n");

    // ---- end to end ----
    println!("  {:<18} {:>4} {:>8} {:>9} {:>8}", "search path", "ef", "recall", "QPS", "vs f32");
    let (mut f64qps, mut i8qps, mut pqqps) = (0.0f64, 0.0f64, 0.0f64);
    for ef in [32usize, 64, 128] {
        let t = Instant::now();
        let base: Vec<Vec<u32>> = (0..NQ).map(|i|
            idx.search(&queries[i*DIM..(i+1)*DIM], K, ef).into_iter().map(|(a,_)|a).collect()).collect();
        let fq = NQ as f64/t.elapsed().as_secs_f64();
        println!("  {:<18} {ef:>4} {:>8.3} {fq:>9.0} {:>8}", "f32 exact", recall(&base,&truth), "1.00x");
        if ef == 64 { f64qps = fq; }

        let t = Instant::now();
        let g8: Vec<Vec<u32>> = (0..NQ).map(|i|
            idx.search_q8(&queries[i*DIM..(i+1)*DIM], K, ef, 4).into_iter().map(|(a,_)|a).collect()).collect();
        let q = NQ as f64/t.elapsed().as_secs_f64();
        println!("  {:<18} {ef:>4} {:>8.3} {q:>9.0} {:>7.2}x", "int8 + rerank 4", recall(&g8,&truth), q/fq);
        if ef == 64 { i8qps = q; }

        for r in [4usize, 16] {
            let t = Instant::now();
            let gp: Vec<Vec<u32>> = (0..NQ).map(|i|
                idx.search_pq(&queries[i*DIM..(i+1)*DIM], K, ef, r).into_iter().map(|(a,_)|a).collect()).collect();
            let q = NQ as f64/t.elapsed().as_secs_f64();
            println!("  {:<18} {ef:>4} {:>8.3} {q:>9.0} {:>7.2}x",
                     format!("pq32 + rerank {r}"), recall(&gp,&truth), q/fq);
            if ef == 64 && r == 4 { pqqps = q; }
        }
    }

    // ---- is this system kernel-bound or memory-bound? ----
    //
    // A 32x smaller payload that does not make search faster demands an
    // explanation, and "cache effects" is not one. Falsifiable version: if the
    // search is bound by the distance kernel, then end-to-end speedup should
    // equal the kernel's ns/call ratio. If it is bound by memory traffic,
    // pq32 (32 B/vector) should pull away from int8 (256 B/vector) and beat
    // that prediction. Printing both is what settles it.
    println!("\n  model check at ef=64 -- predicted from kernel ns/call alone:");
    println!("  {:<16} {:>10} {:>10}", "", "predicted", "actual");
    println!("  {:<16} {:>9.2}x {:>9.2}x", "int8", a / b, i8qps / f64qps);
    println!("  {:<16} {:>9.2}x {:>9.2}x", "pq32", a / c, pqqps / f64qps);
    println!("  payload per vector: f32 1024 B, int8 256 B, pq32 32 B --");
    println!("  if memory traffic were the constraint, pq32 would beat its prediction.");
}

fn main() {
    println!("threads = {}", rayon::current_num_threads());
    for n in [50_000usize, 200_000] { run(n); }
}
