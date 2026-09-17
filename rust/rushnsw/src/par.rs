//! Parallel HNSW construction.
//!
//! WHY THIS FILE EXISTS
//! --------------------
//! The sequential builder's hot signature is `fn insert(&mut self, v: &[f32])`.
//! Wrap that in `par_iter().for_each(|v| idx.insert(v))` and rustc rejects it —
//! not with a vague warning, with a hard error: `&mut Hnsw` is not `Sync`, and
//! two threads cannot hold it. The compiler is right. Two inserts genuinely do
//! race: both append to `data`, both rewrite neighbour lists.
//!
//! The fix is not to fight the checker, it is to notice the checker described
//! the data layout wrong. Insert touches two things with completely different
//! access patterns:
//!
//!   1. `data` — written once, then read by every thread forever.
//!   2. adjacency — read AND written by every thread, at scattered indices.
//!
//! Lumping them in one `&mut self` forces the strictest rule over both. So we
//! split them: `data` is filled up front and becomes a plain `&[f32]` (shared
//! immutable — `Sync` for free, no locks, no atomics, zero cost). Only the
//! adjacency needs interior mutability, and it gets `UnsafeCell` + a *sharded*
//! lock table: `n` locks would be 50k mutexes, one lock would serialise the
//! build, so 4096 shards indexed by `id & mask` gives ~0 contention at 4 cores
//! and costs 4096 * 8 bytes.
//!
//! That is the recurring Rust systems lesson: `&mut self` on a big struct is an
//! over-approximation, and splitting the struct along its real access pattern
//! is both what unblocks the compiler and what makes it fast.
//!
//! SAFETY CONTRACT (what `unsafe impl Sync` is promising)
//! ------------------------------------------------------
//! * `data` and `levels` are never mutated after `build` starts.
//! * every read or write of `level0`/`deg0`/`upper[i]` happens while holding
//!   `locks[i & MASK]` — read guard for reads, write guard for writes.
//! * a node's own lists are published before any other node links back to it,
//!   and a node is only ever *reachable* through an edge, so a half-built node
//!   is invisible to concurrent searchers rather than visible-and-wrong.
//! * no thread ever holds two shard locks at once => no lock-order deadlock.

use crate::dist::{best_l2sq, DistFn};
use crate::hnsw::{Cand, Hnsw, Rev};
use rayon::prelude::*;
use std::cell::UnsafeCell;
use std::collections::BinaryHeap;
use std::sync::RwLock;

const SHARDS: usize = 4096;
const MASK: usize = SHARDS - 1;
const NONE: u32 = u32::MAX;

/// splitmix64 — stateless per-id level sampling. A shared RNG would be a second
/// contention point AND would make the build non-deterministic across thread
/// counts; deriving the level from `(seed, id)` makes a 4-thread build produce
/// the exact same level assignment as a 1-thread build.
fn level_of(seed: u64, id: u32, ml: f64) -> u8 {
    let mut z = seed
        .wrapping_add(0x9E3779B97F4A7C15u64.wrapping_mul(id as u64 + 1));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    let u = ((z >> 11) as f64) / ((1u64 << 53) as f64);
    ((-u.max(1e-12).ln() * ml).floor() as usize).min(16) as u8
}

struct Shared {
    dim: usize,
    m: usize,
    m0: usize,
    ef_c: usize,
    n: usize,
    data: Vec<f32>,
    levels: Vec<u8>,
    dist: DistFn,

    level0: UnsafeCell<Vec<u32>>,
    deg0: UnsafeCell<Vec<u16>>,
    upper: UnsafeCell<Vec<Vec<Vec<u32>>>>,
    locks: Vec<RwLock<()>>,

    entry: RwLock<(u32, u8)>,
}

// The whole safety argument is the module header. This line is the place where
// we take responsibility for it; everything below must keep the contract.
unsafe impl Sync for Shared {}

impl Shared {
    #[inline]
    fn vec_at(&self, id: u32) -> &[f32] {
        let s = id as usize * self.dim;
        &self.data[s..s + self.dim]
    }

    /// Copy a neighbour list out under the shard read lock. The copy is not a
    /// wart — it is the only way to stop holding the lock while we spend 700ns
    /// doing distance calls on those neighbours.
    fn read_nbrs(&self, id: u32, level: usize, out: &mut Vec<u32>) {
        let _g = self.locks[id as usize & MASK].read().unwrap();
        out.clear();
        unsafe {
            if level == 0 {
                let l0 = &*self.level0.get();
                let d = (&*self.deg0.get())[id as usize] as usize;
                let s = id as usize * self.m0;
                out.extend_from_slice(&l0[s..s + d]);
            } else {
                let up = &*self.upper.get();
                let lists = &up[id as usize];
                if level - 1 < lists.len() {
                    out.extend_from_slice(&lists[level - 1]);
                }
            }
        }
    }

    /// Caller MUST hold the write guard for `id`'s shard.
    unsafe fn write_nbrs_locked(&self, id: u32, level: usize, list: &[u32]) {
        unsafe {
            if level == 0 {
                let l0 = &mut *self.level0.get();
                let k = list.len().min(self.m0);
                let s = id as usize * self.m0;
                l0[s..s + k].copy_from_slice(&list[..k]);
                (&mut *self.deg0.get())[id as usize] = k as u16;
            } else {
                let up = &mut *self.upper.get();
                up[id as usize][level - 1] = list.to_vec();
            }
        }
    }
}

/// Per-thread scratch. Allocated once per rayon worker, not once per insert:
/// `visited` is `n * 4` bytes, so per-insert allocation would dominate.
struct Scratch {
    visited: Vec<u32>,
    epoch: u32,
    nbrs: Vec<u32>,
    cur: Vec<u32>,
    calls: u64,
}
impl Scratch {
    fn new(n: usize) -> Self {
        Scratch { visited: vec![0; n], epoch: 0, nbrs: Vec::new(), cur: Vec::new(), calls: 0 }
    }
}

fn search_layer(s: &Shared, sc: &mut Scratch, q: &[f32], eps: &[u32], ef: usize, level: usize)
    -> Vec<Cand>
{
    sc.epoch += 1;
    let ep = sc.epoch;
    let mut cand: BinaryHeap<Rev> = BinaryHeap::with_capacity(ef * 2);
    let mut res: BinaryHeap<Cand> = BinaryHeap::with_capacity(ef + 1);

    for &e in eps {
        if e == NONE || sc.visited[e as usize] == ep { continue; }
        sc.visited[e as usize] = ep;
        sc.calls += 1;
        let d = (s.dist)(q, s.vec_at(e));
        cand.push(Rev(Cand { d, id: e }));
        res.push(Cand { d, id: e });
    }
    while res.len() > ef { res.pop(); }

    let mut nbrs = std::mem::take(&mut sc.nbrs);
    while let Some(Rev(c)) = cand.pop() {
        let worst = res.peek().map(|x| x.d).unwrap_or(f32::INFINITY);
        if c.d > worst && res.len() >= ef { break; }
        s.read_nbrs(c.id, level, &mut nbrs);
        for k in 0..nbrs.len() {
            let nb = nbrs[k];
            if nb == NONE || sc.visited[nb as usize] == ep { continue; }
            sc.visited[nb as usize] = ep;
            let worst = res.peek().map(|x| x.d).unwrap_or(f32::INFINITY);
            sc.calls += 1;
            let d = (s.dist)(q, s.vec_at(nb));
            if res.len() < ef || d < worst {
                cand.push(Rev(Cand { d, id: nb }));
                res.push(Cand { d, id: nb });
                if res.len() > ef { res.pop(); }
            }
        }
    }
    sc.nbrs = nbrs;
    res.into_sorted_vec()
}

/// Alg. 4 pruning. Reads only `data`, which is immutable here — so this needs
/// no lock at all, which is exactly why `data` was hoisted out of the cell.
fn select(s: &Shared, mut pool: Vec<Cand>, m: usize) -> Vec<u32> {
    pool.sort_unstable();
    let mut kept: Vec<Cand> = Vec::with_capacity(m);
    for c in pool {
        if kept.len() >= m { break; }
        let v = s.vec_at(c.id);
        if kept.iter().all(|k| (s.dist)(v, s.vec_at(k.id)) > c.d) {
            kept.push(c);
        }
    }
    kept.into_iter().map(|c| c.id).collect()
}

fn insert_one(s: &Shared, sc: &mut Scratch, id: u32) {
    let q = s.vec_at(id);
    let l = s.levels[id as usize] as usize;

    let (epid, top) = { let g = s.entry.read().unwrap(); (g.0, g.1 as usize) };
    if epid == NONE { return; }

    let mut ep = vec![epid];
    for lev in (l + 1..=top).rev() {
        let r = search_layer(s, sc, q, &ep, 1, lev);
        if r.is_empty() { continue; }
        ep = vec![r[0].id];
    }

    for lev in (0..=l.min(top)).rev() {
        let found = search_layer(s, sc, q, &ep, s.ef_c, lev);
        if found.is_empty() { continue; }
        ep = found.iter().map(|c| c.id).collect();
        let m = if lev == 0 { s.m0 } else { s.m };
        let chosen = select(s, found, m);

        // Publish OUR list first, under our own lock, before anyone links back.
        {
            let _g = s.locks[id as usize & MASK].write().unwrap();
            unsafe { s.write_nbrs_locked(id, lev, &chosen) };
        }

        // Back-edges. Read-modify-write of a neighbour's list must be atomic
        // w.r.t. other threads doing the same to that neighbour, so the whole
        // thing happens under that neighbour's write guard. We hold exactly one
        // lock, and `select` takes none — no lock ordering, no deadlock.
        let mut cur = std::mem::take(&mut sc.cur);
        for &nb in &chosen {
            if nb == id { continue; }
            let _g = s.locks[nb as usize & MASK].write().unwrap();
            cur.clear();
            unsafe {
                if lev == 0 {
                    let l0 = &*s.level0.get();
                    let d = (&*s.deg0.get())[nb as usize] as usize;
                    let st = nb as usize * s.m0;
                    cur.extend_from_slice(&l0[st..st + d]);
                } else {
                    let up = &*s.upper.get();
                    if lev - 1 < up[nb as usize].len() {
                        cur.extend_from_slice(&up[nb as usize][lev - 1]);
                    }
                }
            }
            if cur.contains(&id) { continue; }
            if cur.len() < m {
                cur.push(id);
                unsafe { s.write_nbrs_locked(nb, lev, &cur) };
            } else {
                let nv = s.vec_at(nb);
                let pool: Vec<Cand> = cur
                    .iter()
                    .chain(std::iter::once(&id))
                    .map(|&x| Cand { d: (s.dist)(nv, s.vec_at(x)), id: x })
                    .collect();
                let keep = select(s, pool, m);
                unsafe { s.write_nbrs_locked(nb, lev, &keep) };
            }
        }
        sc.cur = cur;
    }

    if l > 0 {
        let mut g = s.entry.write().unwrap();
        if l as u8 > g.1 { *g = (id, l as u8); }
    }
}

/// Build the whole index from a contiguous `n * dim` buffer, in parallel.
pub fn build(data: Vec<f32>, dim: usize, m: usize, ef_c: usize, seed: u64) -> Hnsw {
    let n = data.len() / dim;
    let m0 = m * 2;
    let ml = 1.0 / (m as f64).ln();
    let levels: Vec<u8> = (0..n as u32).map(|i| level_of(seed, i, ml)).collect();
    let upper: Vec<Vec<Vec<u32>>> =
        levels.iter().map(|&l| vec![Vec::new(); l as usize]).collect();
    let (dist, _) = best_l2sq();

    // Node 0 is the bootstrap entry point: until it exists there is nothing to
    // search, so it is inserted before the parallel region rather than special-
    // cased inside it.
    let s = Shared {
        dim, m, m0, ef_c, n, data, dist,
        level0: UnsafeCell::new(vec![NONE; n * m0]),
        deg0: UnsafeCell::new(vec![0u16; n]),
        upper: UnsafeCell::new(upper),
        locks: (0..SHARDS).map(|_| RwLock::new(())).collect(),
        entry: RwLock::new((0u32, levels[0])),
        levels,
    };

    (1..n as u32).into_par_iter().for_each_init(
        || Scratch::new(n),
        |sc, id| insert_one(&s, sc, id),
    );

    let (entry, max_level) = *s.entry.read().unwrap();
    Hnsw::from_parts(
        dim, m, ef_c,
        s.data,
        s.level0.into_inner(),
        s.deg0.into_inner(),
        s.upper.into_inner(),
        s.levels,
        entry, max_level,
    )
}
