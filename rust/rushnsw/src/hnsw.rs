//! HNSW from scratch. Malkov & Yashunin, arXiv:1603.09320.
//!
//! Layout choices matter more than the algorithm:
//!   - vectors in ONE flat Vec<f32>, stride=dim. Vec<Vec<f32>> would mean a
//!     pointer chase + cache miss per distance call.
//!   - level-0 adjacency in ONE flat Vec<u32>, stride=m0. Same reason.
//!   - visited-set is an epoch-stamped Vec<u32>, not a HashSet: O(1) clear by
//!     bumping a counter, zero allocation per query.

use crate::dist::{best_l2sq, DistFn};
use crate::pq::{best_adc, AdcFn, Pq};
use crate::q8::{best_l2sq_i8, DistI8, Q8};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// f32 has no Ord (NaN). Newtype + explicit total order is the idiomatic escape.
#[derive(Copy, Clone, PartialEq)]
pub(crate) struct Cand {
    pub(crate) d: f32,
    pub(crate) id: u32,
}
impl Eq for Cand {}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        self.d.partial_cmp(&o.d).unwrap_or(Ordering::Equal)
    }
}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
/// Reversed order => BinaryHeap (a max-heap) behaves as a min-heap.
#[derive(Copy, Clone, PartialEq, Eq)]
pub(crate) struct Rev(pub(crate) Cand);
impl Ord for Rev {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.cmp(&self.0)
    }
}
impl PartialOrd for Rev {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// xorshift64*. Deterministic, no dependencies, good enough for level sampling.
struct Rng(u64);
impl Rng {
    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        ((x.wrapping_mul(0x2545F491_4F6CDD1D) >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

pub struct Hnsw {
    dim: usize,
    m: usize,
    m0: usize,
    ef_construction: usize,
    ml: f64,

    data: Vec<f32>,          // n * dim
    level0: Vec<u32>,        // n * m0
    deg0: Vec<u16>,          // n
    upper: Vec<Vec<Vec<u32>>>, // node -> level-1 -> neighbors
    levels: Vec<u8>,         // n

    entry: u32,
    max_level: u8,
    n: usize,

    dist: DistFn,
    pub kernel: &'static str,
    q8: Option<(Q8, DistI8)>,
    pq: Option<(Pq, AdcFn)>,
    rng: Rng,

    // query scratch, reused across calls
    visited: Vec<u32>,
    epoch: u32,
    pub dist_calls: u64,
}

const NONE: u32 = u32::MAX;

impl Hnsw {
    pub fn new(dim: usize, m: usize, ef_construction: usize, seed: u64) -> Self {
        let (dist, kernel) = best_l2sq();
        Self {
            dim,
            m,
            m0: m * 2,
            ef_construction,
            ml: 1.0 / (m as f64).ln(),
            data: Vec::new(),
            level0: Vec::new(),
            deg0: Vec::new(),
            upper: Vec::new(),
            levels: Vec::new(),
            entry: NONE,
            max_level: 0,
            n: 0,
            dist,
            kernel,
            q8: None,
            pq: None,
            rng: Rng(seed | 1),
            visited: Vec::new(),
            epoch: 0,
            dist_calls: 0,
        }
    }

    #[inline]
    fn vec_at(&self, id: u32) -> &[f32] {
        let s = id as usize * self.dim;
        &self.data[s..s + self.dim]
    }

    #[inline]
    fn d(&mut self, q: &[f32], id: u32) -> f32 {
        self.dist_calls += 1;
        let s = id as usize * self.dim;
        (self.dist)(q, &self.data[s..s + self.dim])
    }

    #[inline]
    fn nbrs0(&self, id: u32) -> &[u32] {
        let s = id as usize * self.m0;
        &self.level0[s..s + self.deg0[id as usize] as usize]
    }

    fn nbrs(&self, id: u32, level: usize) -> &[u32] {
        if level == 0 {
            self.nbrs0(id)
        } else {
            &self.upper[id as usize][level - 1]
        }
    }

    /// Greedy beam search on one layer. This is the whole engine.
    /// Invariant: `res` is a max-heap of the ef best found so far; `cand` is a
    /// min-heap of the frontier. Stop when the nearest frontier node is worse
    /// than the worst result — nothing closer can be reached.
    fn search_layer(&mut self, q: &[f32], eps: &[u32], ef: usize, level: usize) -> Vec<Cand> {
        self.epoch += 1;
        let ep = self.epoch;
        if self.visited.len() < self.n {
            self.visited.resize(self.n, 0);
        }

        let mut cand: BinaryHeap<Rev> = BinaryHeap::with_capacity(ef * 2);
        let mut res: BinaryHeap<Cand> = BinaryHeap::with_capacity(ef + 1);

        for &e in eps {
            if self.visited[e as usize] == ep {
                continue;
            }
            self.visited[e as usize] = ep;
            let d = self.d(q, e);
            cand.push(Rev(Cand { d, id: e }));
            res.push(Cand { d, id: e });
        }
        while res.len() > ef {
            res.pop();
        }

        let mut scratch: Vec<u32> = Vec::with_capacity(self.m0);
        while let Some(Rev(c)) = cand.pop() {
            let worst = res.peek().map(|x| x.d).unwrap_or(f32::INFINITY);
            if c.d > worst && res.len() >= ef {
                break;
            }
            // copy out: borrow of self ends before the &mut self calls below.
            scratch.clear();
            scratch.extend_from_slice(self.nbrs(c.id, level));

            for k in 0..scratch.len() {
                let nb = scratch[k];
                if self.visited[nb as usize] == ep {
                    continue;
                }
                self.visited[nb as usize] = ep;
                let worst = res.peek().map(|x| x.d).unwrap_or(f32::INFINITY);
                let d = self.d(q, nb);
                if res.len() < ef || d < worst {
                    cand.push(Rev(Cand { d, id: nb }));
                    res.push(Cand { d, id: nb });
                    if res.len() > ef {
                        res.pop();
                    }
                }
            }
        }
        res.into_sorted_vec()
    }

    /// Neighbor selection heuristic (Alg. 4). Keep `e` only if it is closer to
    /// the query than to every already-kept neighbor. This is what stops the
    /// graph collapsing into clusters and keeps long-range "highway" edges —
    /// the single biggest recall lever in HNSW.
    /// NOTE on `&self` vs `&mut self`: this was originally `&mut self` purely by
    /// habit — it mutates nothing. That one wrong word forced `.to_vec()` on
    /// every candidate, because `&self` from `vec_at` cannot outlive a `&mut
    /// self` call, so the fix "looked like" copying the vector out. Narrowing
    /// the receiver to `&self` deletes an allocation per candidate, i.e. ~m0
    /// heap allocations per insert. The borrow checker was never the problem;
    /// an over-broad receiver was.
    fn select(&self, base: &[f32], mut pool: Vec<Cand>, m: usize) -> Vec<u32> {
        pool.sort_unstable();
        let mut kept: Vec<Cand> = Vec::with_capacity(m);
        for c in pool {
            if kept.len() >= m {
                break;
            }
            let v = self.vec_at(c.id);
            if kept.iter().all(|k| (self.dist)(v, self.vec_at(k.id)) > c.d) {
                kept.push(c);
            }
        }
        let _ = base;
        kept.into_iter().map(|c| c.id).collect()
    }

    fn set_nbrs(&mut self, id: u32, level: usize, list: &[u32]) {
        if level == 0 {
            let s = id as usize * self.m0;
            let n = list.len().min(self.m0);
            self.level0[s..s + n].copy_from_slice(&list[..n]);
            self.deg0[id as usize] = n as u16;
        } else {
            self.upper[id as usize][level - 1] = list.to_vec();
        }
    }

    pub fn insert(&mut self, v: &[f32]) -> u32 {
        assert_eq!(v.len(), self.dim);
        let id = self.n as u32;
        self.data.extend_from_slice(v);

        let l = (-self.rng.next_f64().max(1e-12).ln() * self.ml).floor() as usize;
        let l = l.min(16);
        self.levels.push(l as u8);
        self.level0.resize(self.level0.len() + self.m0, NONE);
        self.deg0.push(0);
        self.upper.push(vec![Vec::new(); l]);
        self.n += 1;

        if self.entry == NONE {
            self.entry = id;
            self.max_level = l as u8;
            return id;
        }

        let q = v.to_vec();
        let mut ep = vec![self.entry];
        let top = self.max_level as usize;

        // Phase 1: zoom in from the top with a width-1 greedy walk.
        for lev in (l + 1..=top).rev() {
            let r = self.search_layer(&q, &ep, 1, lev);
            ep = vec![r[0].id];
        }

        // Phase 2: at each layer we join, find ef_construction candidates,
        // prune to m, and wire both directions.
        for lev in (0..=l.min(top)).rev() {
            let found = self.search_layer(&q, &ep, self.ef_construction, lev);
            ep = found.iter().map(|c| c.id).collect();
            let m = if lev == 0 { self.m0 } else { self.m };
            let chosen = self.select(&q, found, m);
            self.set_nbrs(id, lev, &chosen);

            for &nb in &chosen {
                let mut cur: Vec<u32> = self.nbrs(nb, lev).to_vec();
                if cur.len() < m {
                    cur.push(id);
                    self.set_nbrs(nb, lev, &cur);
                } else {
                    // Over capacity: re-run the heuristic over old+new.
                    let nv = self.vec_at(nb).to_vec();
                    let mut pool: Vec<Cand> = cur
                        .iter()
                        .chain(std::iter::once(&id))
                        .map(|&x| Cand { d: (self.dist)(&nv, self.vec_at(x)), id: x })
                        .collect();
                    let keep = self.select(&nv, pool, m);
                    self.set_nbrs(nb, lev, &keep);
                }
            }
        }

        if l as u8 > self.max_level {
            self.max_level = l as u8;
            self.entry = id;
        }
        id
    }

    pub fn search(&mut self, q: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        if self.entry == NONE {
            return Vec::new();
        }
        let mut ep = vec![self.entry];
        for lev in (1..=self.max_level as usize).rev() {
            let r = self.search_layer(q, &ep, 1, lev);
            ep = vec![r[0].id];
        }
        let r = self.search_layer(q, &ep, ef.max(k), 0);
        r.into_iter().take(k).map(|c| (c.id, c.d)).collect()
    }

    /// Adopt a graph built elsewhere (see `par::build`). The parallel builder
    /// owns a different memory layout while it is running — `UnsafeCell` +
    /// sharded locks — and hands the finished, now-immutable arrays over here
    /// so there is exactly one query path in the crate.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        dim: usize, m: usize, ef_construction: usize,
        data: Vec<f32>, level0: Vec<u32>, deg0: Vec<u16>,
        upper: Vec<Vec<Vec<u32>>>, levels: Vec<u8>,
        entry: u32, max_level: u8,
    ) -> Self {
        let (dist, kernel) = best_l2sq();
        let n = levels.len();
        Self {
            dim, m, m0: m * 2, ef_construction, ml: 1.0 / (m as f64).ln(),
            data, level0, deg0, upper, levels, entry, max_level, n,
            dist, kernel, q8: None, pq: None, rng: Rng(1),
            visited: Vec::new(), epoch: 0, dist_calls: 0,
        }
    }

    /// Graph health: mean/min out-degree at level 0, and how many nodes are
    /// unreachable-ish (degree 0). A parallel build that races will show up
    /// here before it shows up in recall.
    pub fn degree_stats(&self) -> (f64, u16, usize) {
        let sum: u64 = self.deg0.iter().map(|&d| d as u64).sum();
        let min = self.deg0.iter().copied().min().unwrap_or(0);
        let orphans = self.deg0.iter().filter(|&&d| d == 0).count();
        (sum as f64 / self.n as f64, min, orphans)
    }

    /// Build the int8 view of the corpus. Traversal then runs entirely on codes
    /// (12.8 MB instead of 51 MB at 50k x 256) and only the final shortlist is
    /// rescored in f32. This split — cheap approximate traversal, exact rerank —
    /// is how every production vector DB actually spends its memory bandwidth.
    pub fn quantize(&mut self) {
        let q = Q8::from_f32(&self.data, self.dim);
        let (f, _) = best_l2sq_i8();
        self.q8 = Some((q, f));
    }

    pub fn q8_bytes(&self) -> usize {
        self.q8.as_ref().map(|(q, _)| q.bytes()).unwrap_or(0)
    }
    pub fn f32_bytes(&self) -> usize { self.data.len() * 4 }
    pub fn graph_bytes(&self) -> usize {
        self.level0.len() * 4
            + self.deg0.len() * 2
            + self.upper.iter().flatten().map(|v| v.len() * 4).sum::<usize>()
    }

    /// Same beam search, integer distances. `d` is stored as f32 only because
    /// `Cand` is shared with the float path — the COMPARISONS are all integer,
    /// and i32 -> f32 is exact below 2^24, which every distance here is.
    fn search_layer_q8(&mut self, qc: &[i8], eps: &[u32], ef: usize, level: usize) -> Vec<Cand> {
        // Classic `&self` field vs `&mut self` method conflict: the loop needs
        // `&q8` alive across calls that take `&mut self` (visited, dist_calls).
        // The unsafe raw-pointer "borrow split" works, but `Option::take` is the
        // safe idiom and compiles to the same thing — move the field out, run,
        // move it back. The type system now enforces what the comment would
        // otherwise only assert.
        let owned = self.q8.take().expect("quantize() first");
        let (q8, f) = (&owned.0, owned.1);
        self.epoch += 1;
        let ep = self.epoch;
        if self.visited.len() < self.n { self.visited.resize(self.n, 0); }

        let mut cand: BinaryHeap<Rev> = BinaryHeap::with_capacity(ef * 2);
        let mut res: BinaryHeap<Cand> = BinaryHeap::with_capacity(ef + 1);
        for &e in eps {
            if self.visited[e as usize] == ep { continue; }
            self.visited[e as usize] = ep;
            self.dist_calls += 1;
            let d = f(qc, q8.at(e)) as f32;
            cand.push(Rev(Cand { d, id: e }));
            res.push(Cand { d, id: e });
        }
        while res.len() > ef { res.pop(); }

        let mut scratch: Vec<u32> = Vec::with_capacity(self.m0);
        while let Some(Rev(c)) = cand.pop() {
            let worst = res.peek().map(|x| x.d).unwrap_or(f32::INFINITY);
            if c.d > worst && res.len() >= ef { break; }
            scratch.clear();
            scratch.extend_from_slice(self.nbrs(c.id, level));
            for k in 0..scratch.len() {
                let nb = scratch[k];
                if self.visited[nb as usize] == ep { continue; }
                self.visited[nb as usize] = ep;
                let worst = res.peek().map(|x| x.d).unwrap_or(f32::INFINITY);
                self.dist_calls += 1;
                let d = f(qc, q8.at(nb)) as f32;
                if res.len() < ef || d < worst {
                    cand.push(Rev(Cand { d, id: nb }));
                    res.push(Cand { d, id: nb });
                    if res.len() > ef { res.pop(); }
                }
            }
        }
        self.q8 = Some(owned);
        res.into_sorted_vec()
    }

    /// Quantized search with exact rerank. `rerank` is a multiplier on k: pull
    /// `k * rerank` candidates using int8, rescore those in f32, return the best
    /// k. rerank = 1 means no rescoring at all.
    pub fn search_q8(&mut self, q: &[f32], k: usize, ef: usize, rerank: usize) -> Vec<(u32, f32)> {
        if self.entry == NONE || self.q8.is_none() { return Vec::new(); }
        let qc = self.q8.as_ref().unwrap().0.encode(q);
        let mut ep = vec![self.entry];
        for lev in (1..=self.max_level as usize).rev() {
            let r = self.search_layer_q8(&qc, &ep, 1, lev);
            ep = vec![r[0].id];
        }
        let shortlist = self.search_layer_q8(&qc, &ep, ef.max(k * rerank), 0);
        if rerank <= 1 {
            let q8 = self.q8.as_ref().unwrap().0.scale;
            return shortlist.into_iter().take(k)
                .map(|c| (c.id, c.d * q8 * q8)).collect();
        }
        let mut out: Vec<(u32, f32)> = shortlist.into_iter()
            .take(k * rerank)
            .map(|c| {
                let s = c.id as usize * self.dim;
                (c.id, (self.dist)(q, &self.data[s..s + self.dim]))
            })
            .collect();
        out.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        out.truncate(k);
        out
    }

    /// Cheap structural hash of the level assignment. Used by tests to prove the
    /// build does not depend on thread count.
    pub fn levels_fingerprint(&self) -> u64 {
        self.levels.iter().fold(1469598103934665603u64, |h, &l| {
            (h ^ l as u64).wrapping_mul(1099511628211)
        })
    }

    /// Train the product quantizer over the corpus and attach it.
    /// `m` subquantizers => `m` bytes per vector (dim*4 / m compression).
    pub fn train_pq(&mut self, m: usize, iters: usize, max_train: usize, seed: u64) {
        let p = Pq::train(&self.data, self.dim, m, iters, max_train, seed);
        let (f, _) = best_adc();
        self.pq = Some((p, f));
    }

    pub fn pq_bytes(&self) -> usize { self.pq.as_ref().map(|(p, _)| p.bytes()).unwrap_or(0) }
    pub fn pq_codebook_bytes(&self) -> usize {
        self.pq.as_ref().map(|(p, _)| p.codebook_bytes()).unwrap_or(0)
    }
    pub fn pq_lut_bytes(&self) -> usize {
        self.pq.as_ref().map(|(p, _)| p.lut_bytes()).unwrap_or(0)
    }

    /// Beam search where the distance is a table lookup.
    ///
    /// Note what is NOT here: no query vector in the inner loop, no corpus
    /// vector, no multiply. `lut` was built once before the descent and every
    /// node costs `m` gathers. The graph traversal is identical; only the
    /// oracle changed.
    fn search_layer_pq(&mut self, lut: &[f32], eps: &[u32], ef: usize, level: usize) -> Vec<Cand> {
        let owned = self.pq.take().expect("train_pq() first");
        let (pq, adc) = (&owned.0, owned.1);
        self.epoch += 1;
        let ep = self.epoch;
        if self.visited.len() < self.n { self.visited.resize(self.n, 0); }

        let mut cand: BinaryHeap<Rev> = BinaryHeap::with_capacity(ef * 2);
        let mut res: BinaryHeap<Cand> = BinaryHeap::with_capacity(ef + 1);
        for &e in eps {
            if self.visited[e as usize] == ep { continue; }
            self.visited[e as usize] = ep;
            self.dist_calls += 1;
            let d = adc(lut, pq.code(e), pq.ksub);
            cand.push(Rev(Cand { d, id: e }));
            res.push(Cand { d, id: e });
        }
        while res.len() > ef { res.pop(); }

        let mut scratch: Vec<u32> = Vec::with_capacity(self.m0);
        while let Some(Rev(c)) = cand.pop() {
            let worst = res.peek().map(|x| x.d).unwrap_or(f32::INFINITY);
            if c.d > worst && res.len() >= ef { break; }
            scratch.clear();
            scratch.extend_from_slice(self.nbrs(c.id, level));
            for k in 0..scratch.len() {
                let nb = scratch[k];
                if self.visited[nb as usize] == ep { continue; }
                self.visited[nb as usize] = ep;
                let worst = res.peek().map(|x| x.d).unwrap_or(f32::INFINITY);
                self.dist_calls += 1;
                let d = adc(lut, pq.code(nb), pq.ksub);
                if res.len() < ef || d < worst {
                    cand.push(Rev(Cand { d, id: nb }));
                    res.push(Cand { d, id: nb });
                    if res.len() > ef { res.pop(); }
                }
            }
        }
        self.pq = Some(owned);
        res.into_sorted_vec()
    }

    pub fn search_pq(&mut self, q: &[f32], k: usize, ef: usize, rerank: usize) -> Vec<(u32, f32)> {
        if self.entry == NONE || self.pq.is_none() { return Vec::new(); }
        let lut = self.pq.as_ref().unwrap().0.lut(q);
        let mut ep = vec![self.entry];
        for lev in (1..=self.max_level as usize).rev() {
            let r = self.search_layer_pq(&lut, &ep, 1, lev);
            ep = vec![r[0].id];
        }
        let shortlist = self.search_layer_pq(&lut, &ep, ef.max(k * rerank), 0);
        if rerank <= 1 {
            return shortlist.into_iter().take(k).map(|c| (c.id, c.d)).collect();
        }
        let mut out: Vec<(u32, f32)> = shortlist.into_iter().take(k * rerank)
            .map(|c| {
                let s = c.id as usize * self.dim;
                (c.id, (self.dist)(q, &self.data[s..s + self.dim]))
            }).collect();
        out.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        out.truncate(k);
        out
    }

    pub fn len(&self) -> usize { self.n }
    pub fn max_level(&self) -> u8 { self.max_level }
}
