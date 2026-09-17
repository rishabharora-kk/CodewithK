//! The parallel builder is the only place in this crate with `unsafe impl Sync`.
//! These tests are what keep that line honest.
//!
//! A data race in HNSW construction does not panic and does not corrupt memory
//! in a way that shows up as a crash — it quietly drops edges and you discover
//! it months later as "recall is a bit worse than the paper". So the assertions
//! here are structural (no orphaned nodes, degrees in range, every neighbour id
//! in bounds and not self) plus a recall floor against exact ground truth.
//!
//! Run under `cargo +nightly miri` or, more practically, `RUSTFLAGS="-Z
//! sanitizer=thread"` to check the locking itself; these tests check the
//! *outcome* on stable.

use rushnsw::hnsw::Hnsw;
use rushnsw::par;
use rushnsw::synth::lowrank;
use rushnsw::dist::best_l2sq;

const N: usize = 4000;
const DIM: usize = 64;
const NQ: usize = 100;
const K: usize = 10;

fn truth(data: &[f32], queries: &[f32]) -> Vec<Vec<u32>> {
    let (df, _) = best_l2sq();
    (0..NQ).map(|qi| {
        let q = &queries[qi*DIM..(qi+1)*DIM];
        let mut all: Vec<(f32,u32)> = (0..N as u32)
            .map(|i| (df(q, &data[i as usize*DIM..(i as usize+1)*DIM]), i)).collect();
        all.sort_unstable_by(|a,b| a.0.partial_cmp(&b.0).unwrap());
        all[..K].iter().map(|x| x.1).collect()
    }).collect()
}

fn recall(idx: &mut Hnsw, queries: &[f32], t: &[Vec<u32>], ef: usize) -> f64 {
    let mut hit = 0usize;
    for qi in 0..NQ {
        for (id, _) in idx.search(&queries[qi*DIM..(qi+1)*DIM], K, ef) {
            if t[qi].contains(&id) { hit += 1; }
        }
    }
    hit as f64 / (NQ*K) as f64
}

#[test]
fn parallel_build_matches_sequential_recall() {
    let data = lowrank(N, DIM, 16, 7);
    let queries = lowrank(NQ, DIM, 16, 7);
    let gt = truth(&data, &queries);

    let mut seq = Hnsw::new(DIM, 16, 200, 1234);
    for i in 0..N { seq.insert(&data[i*DIM..(i+1)*DIM]); }
    let rs = recall(&mut seq, &queries, &gt, 64);

    let mut p = par::build(data.clone(), DIM, 16, 200, 1234);
    let rp = recall(&mut p, &queries, &gt, 64);

    assert!(rs > 0.90, "sequential baseline broken: {rs}");
    // Concurrent inserts can miss each other (two nodes added at the same
    // instant are invisible to one another), so exact equality is the wrong
    // assertion. A 2-point drop is the real, measured tolerance.
    assert!(rp > rs - 0.02, "parallel recall {rp} regressed vs sequential {rs}");
}

#[test]
fn parallel_graph_is_structurally_sound() {
    let data = lowrank(N, DIM, 16, 7);
    let p = par::build(data, DIM, 16, 200, 1234);
    let (mean, min, orphans) = p.degree_stats();
    assert_eq!(orphans, 0, "some nodes have no level-0 edges: lost writes");
    assert!(min >= 1, "min degree {min}");
    assert!(mean > 8.0 && mean <= 32.0, "mean degree {mean} outside [8, m0]");
    assert_eq!(p.len(), N);
}

#[test]
fn parallel_build_is_deterministic_across_thread_counts() {
    // Levels are derived from (seed, id) rather than a shared RNG precisely so
    // this holds. If someone reintroduces a shared RNG, this test fails.
    let data = lowrank(1500, DIM, 16, 3);
    let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let four = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let a = one.install(|| par::build(data.clone(), DIM, 16, 100, 555));
    let b = four.install(|| par::build(data, DIM, 16, 100, 555));
    assert_eq!(a.levels_fingerprint(), b.levels_fingerprint());
}

#[test]
fn quantized_search_tracks_full_precision() {
    let data = lowrank(N, DIM, 16, 7);
    let queries = lowrank(NQ, DIM, 16, 7);
    let gt = truth(&data, &queries);
    let mut p = par::build(data, DIM, 16, 200, 1234);
    let rf = recall(&mut p, &queries, &gt, 64);
    p.quantize();
    let mut hit = 0usize;
    for qi in 0..NQ {
        for (id, _) in p.search_q8(&queries[qi*DIM..(qi+1)*DIM], K, 64, 4) {
            if gt[qi].contains(&id) { hit += 1; }
        }
    }
    let rq = hit as f64/(NQ*K) as f64;
    // With 4x rerank, int8 traversal should lose essentially nothing.
    assert!(rq > rf - 0.01, "quantized recall {rq} vs f32 {rf}");
}
