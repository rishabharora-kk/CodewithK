//! Round 2: does parallel construction preserve the graph, and does int8 pay?
//!
//! Every number here is measured on THIS machine against exact ground truth.
//! Nothing is quoted from a paper.

use rushnsw::dist::best_l2sq;
use rushnsw::hnsw::Hnsw;
use rushnsw::q8::{best_l2sq_i8, Q8};
use rushnsw::synth::lowrank;
use rushnsw::par;
use std::time::Instant;

const N: usize = 50_000;
const DIM: usize = 256;
const NQ: usize = 500;
const K: usize = 10;
const RANK: usize = 32; // hard-but-realistic regime found in round 1

fn recall(got: &[Vec<u32>], truth: &[Vec<u32>]) -> f64 {
    let mut hit = 0usize;
    for (g, t) in got.iter().zip(truth) {
        for id in g { if t.contains(id) { hit += 1; } }
    }
    hit as f64 / (truth.len() * K) as f64
}

fn sweep(idx: &mut Hnsw, queries: &[f32], truth: &[Vec<u32>], label: &str) {
    println!("  {label:<22} {:>6} {:>9} {:>9}", "ef", "recall", "QPS");
    for ef in [32usize, 64, 128] {
        let t = Instant::now();
        let got: Vec<Vec<u32>> = (0..NQ)
            .map(|i| idx.search(&queries[i*DIM..(i+1)*DIM], K, ef)
                        .into_iter().map(|(id,_)| id).collect())
            .collect();
        let qps = NQ as f64 / t.elapsed().as_secs_f64();
        println!("  {:<22} {ef:>6} {:>9.3} {:>9.0}", "", recall(&got, truth), qps);
    }
}

fn main() {
    let threads = rayon::current_num_threads();
    let (df, fk) = best_l2sq();
    let (di, ik) = best_l2sq_i8();
    println!("n={N} dim={DIM} intrinsic_rank={RANK}  threads={threads}");
    println!("kernels: f32={fk}  i8={ik}\n");

    let data = lowrank(N, DIM, RANK, 99);
    let queries = lowrank(NQ, DIM, RANK, 99);

    // ---------- 1. kernel: float vs int8 ----------
    println!("== distance kernel, {DIM}d ==");
    let a = &data[..DIM];
    let b = &data[DIM..2*DIM];
    let q8probe = Q8::from_f32(&data, DIM);
    let (ca, cb) = (q8probe.at(0).to_vec(), q8probe.at(1).to_vec());
    let iters = 5_000_000u64;

    let t = Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..iters { acc += df(a, b); }
    let fns = t.elapsed().as_nanos() as f64 / iters as f64;

    let t = Instant::now();
    let mut acci = 0i64;
    for _ in 0..iters { acci += di(&ca, &cb) as i64; }
    let ins = t.elapsed().as_nanos() as f64 / iters as f64;
    std::hint::black_box((acc, acci));

    println!("  f32 {fk:<11} {fns:6.1} ns   {:5.1} GB/s", 2.0*DIM as f64*4.0/fns);
    println!("  i8  {ik:<11} {ins:6.1} ns   {:5.1} GB/s   {:.2}x faster\n",
             2.0*DIM as f64/ins, fns/ins);

    // ---------- 2. exact ground truth ----------
    let t = Instant::now();
    let truth: Vec<Vec<u32>> = (0..NQ).map(|qi| {
        let q = &queries[qi*DIM..(qi+1)*DIM];
        let mut all: Vec<(f32,u32)> = (0..N as u32)
            .map(|i| (df(q, &data[i as usize*DIM..(i as usize+1)*DIM]), i)).collect();
        all.sort_unstable_by(|x,y| x.0.partial_cmp(&y.0).unwrap());
        all[..K].iter().map(|x| x.1).collect()
    }).collect();
    let bf_qps = NQ as f64 / t.elapsed().as_secs_f64();
    println!("== brute force f32 ==\n  {bf_qps:.1} QPS (exact, this is ground truth)\n");

    // ---------- 3. build: sequential vs parallel ----------
    println!("== build ==");
    let t = Instant::now();
    let mut seq = Hnsw::new(DIM, 16, 200, 1234);
    for i in 0..N { seq.insert(&data[i*DIM..(i+1)*DIM]); }
    let tseq = t.elapsed();
    let (dseq, minseq, orphseq) = seq.degree_stats();
    println!("  sequential  {:>7.2?}  {:>7.0}/s  deg0 mean {dseq:.1} min {minseq} orphans {orphseq}",
             tseq, N as f64/tseq.as_secs_f64());

    // Thread sweep. The 1-thread row is the control: it runs the SAME concurrent
    // code path, same locks, same rayon range-splitting, on one core. Anything
    // the 1-thread build gains over `sequential` is NOT parallelism — it is the
    // rewrite. Without this row a 4.3x "speedup on 4 cores" is unfalsifiable.
    let mut parx = None;
    let mut t1 = tseq;
    for th in [1usize, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(th).build().unwrap();
        let d = data.clone();
        let t = Instant::now();
        let idx = pool.install(|| par::build(d, DIM, 16, 200, 1234));
        let el = t.elapsed();
        if th == 1 { t1 = el; }
        let (dp, mn, orph) = idx.degree_stats();
        println!("  parallel/{th:<2}  {:>7.2?}  {:>7.0}/s  deg0 mean {dp:.1} min {mn} orphans {orph}  \
vs-seq {:.2}x  vs-1thread {:.2}x",
                 el, N as f64/el.as_secs_f64(),
                 tseq.as_secs_f64()/el.as_secs_f64(), t1.as_secs_f64()/el.as_secs_f64());
        if th == threads.min(4) { parx = Some(idx); }
    }
    let mut parx = parx.unwrap();
    println!("  -> vs-1thread is the honest scaling number; vs-seq folds in the rewrite.\n");

    // ---------- 4. did the race cost recall? ----------
    println!("== recall@{K} vs exact ==");
    sweep(&mut seq,  &queries, &truth, "sequential build");
    sweep(&mut parx, &queries, &truth, "parallel build");
    println!();

    // ---------- 5. what int8 costs, isolated from the graph ----------
    let q8 = Q8::from_f32(&data, DIM);
    println!("== int8 quantization ==");
    println!("  f32 corpus {:>8.1} MB", (data.len()*4) as f64/1e6);
    println!("  i8  corpus {:>8.1} MB   ({:.1}x smaller, scale {:.5})",
             q8.bytes() as f64/1e6, (data.len()*4) as f64/q8.bytes() as f64, q8.scale);

    let t = Instant::now();
    let mut hit = 0usize;
    for qi in 0..NQ {
        let qc = q8.encode(&queries[qi*DIM..(qi+1)*DIM]);
        for (id, _) in rushnsw::q8::brute_topk(&q8, &qc, K, di) {
            if truth[qi].contains(&id) { hit += 1; }
        }
    }
    let qps8 = NQ as f64 / t.elapsed().as_secs_f64();
    println!("  brute force i8: recall@{K} {:.4}  {qps8:.1} QPS  ({:.2}x vs f32 brute force)",
             hit as f64/(NQ*K) as f64, qps8/bf_qps);
    println!("  -> that recall gap is the PRICE of quantization alone, graph excluded.");

    // ---------- 6. int8 INSIDE the graph, with exact rerank ----------
    parx.quantize();
    println!("\n== quantized HNSW (traverse on i8, rescore top k*r in f32) ==");
    println!("  memory: graph {:.1} MB + i8 {:.1} MB = {:.1} MB resident",
             parx.graph_bytes() as f64/1e6, parx.q8_bytes() as f64/1e6,
             (parx.graph_bytes()+parx.q8_bytes()) as f64/1e6);
    println!("          (f32 corpus {:.1} MB is only touched during rerank)",
             parx.f32_bytes() as f64/1e6);
    println!("  {:>4} {:>7} {:>9} {:>9} {:>9}", "ef", "rerank", "recall", "QPS", "vs f32");

    // f32 reference QPS at each ef, measured in the same loop shape
    for ef in [32usize, 64, 128] {
        let t = Instant::now();
        let base: Vec<Vec<u32>> = (0..NQ).map(|i| parx.search(&queries[i*DIM..(i+1)*DIM], K, ef)
            .into_iter().map(|(id,_)| id).collect()).collect();
        let fqps = NQ as f64 / t.elapsed().as_secs_f64();
        println!("  {ef:>4} {:>7} {:>9.3} {:>9.0} {:>9}", "f32", recall(&base, &truth), fqps, "1.00x");
        for r in [1usize, 4] {
            let t = Instant::now();
            let got: Vec<Vec<u32>> = (0..NQ).map(|i| parx.search_q8(&queries[i*DIM..(i+1)*DIM], K, ef, r)
                .into_iter().map(|(id,_)| id).collect()).collect();
            let qps = NQ as f64 / t.elapsed().as_secs_f64();
            println!("  {ef:>4} {r:>7} {:>9.3} {:>9.0} {:>8.2}x", recall(&got, &truth), qps, qps/fqps);
        }
    }
}
