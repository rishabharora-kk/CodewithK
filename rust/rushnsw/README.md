# rushnsw

An HNSW approximate-nearest-neighbour index written from the paper
(Malkov & Yashunin, [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)), with
the distance kernels written in raw AVX-512 intrinsics, a lock-sharded parallel
builder, and int8 scalar quantization with exact rerank.

One dependency: `rayon`. Everything else — SIMD, quantization, the graph, the
RNG, the benchmark harness — is in this crate.

```
cargo test  --release          # 6 tests, incl. concurrency + kernel exactness
cargo run   --release --bin bench   # round 1: kernel + recall/ef + intrinsic dim
cargo run   --release --bin lab     # round 2: parallel build + int8
```

---

## Measured results

All numbers below are from `src/bin/lab.rs` on a 4-core AVX-512 (VNNI) box,
n = 50 000, dim = 256, intrinsic rank 32, recall@10 against **exact** brute-force
ground truth. Nothing is quoted from a paper.

### Distance kernels

| kernel | ns/call | vs scalar |
|---|---:|---:|
| `l2sq_scalar` f32 | 277 | 1.0x |
| `l2sq_avx512` f32 | 15.4 | **18x** |
| `l2sq_i8_vnni` int8 | 8.8 | **31x** |

The int8 kernel uses `_mm512_dpwssd_epi32` (AVX512-VNNI): one fused i16
multiply-add where the float path needs a multiply and an add, and 32 elements
per register instead of 16.

Integer kernels admit **exact** tests — `tests` assert `vnni == scalar` bit for
bit, at every dimension and at maximum lane separation. A float kernel can only
ever be checked against an epsilon.

### Parallel construction

| build | wall | vs 1 thread |
|---|---:|---:|
| sequential (`&mut self`) | 23.30 s | — |
| parallel, 1 thread | 22.81 s | 1.00x |
| parallel, 2 threads | 11.40 s | 2.00x |
| parallel, 4 threads | **5.66 s** | **4.03x** |

The 1-thread row is the control, and it is the only reason the 4-thread number
means anything. First measurement showed "4.25x on 4 cores — 106% efficiency",
which is not a result, it is a bug report about the measurement. Two hypotheses:
(a) rayon's range splitting changes insertion order so early inserts see a
sparser, cheaper graph; (b) the rewrite dropped an allocation the sequential
path was paying. Fixing (b) moved nothing. Adding the 1-thread control killed
(a): the concurrent code path on one core is within 2% of the sequential one, so
the speedup is real parallelism and the 106% was timer noise.

Graph quality is preserved, not merely "close": mean level-0 degree 24.4 in both,
**zero orphaned nodes**, recall within 0.005 of the sequential build at every ef.

### int8 quantization

| | f32 | int8 + 4x rerank |
|---|---:|---:|
| resident corpus | 51.2 MB | 12.8 MB |
| index total (graph + corpus) | 57.9 MB | **19.5 MB** |
| recall@10, ef=64 | 0.985 | 0.984 |
| QPS, ef=64 | 5 855 | **11 880** |

Traversal runs entirely on int8 codes; only the top `k x rerank` shortlist is
rescored in f32. At ef=32 the quantized path with 4x rerank actually scores
*higher* recall than f32 (0.968 vs 0.957) — pulling a wider shortlist and
rescoring it is a better use of the same budget than a narrower exact beam.

Quantization's isolated cost, measured on brute force with the graph removed
entirely: recall@10 drops 0.9858 from 1.000. Everything below that in the table
is the graph's fault, not the quantizer's.

---

## What the Rust actually taught me

**`&mut self` is an over-approximation, and that is the whole lesson.**
`par_iter().for_each(|v| idx.insert(v))` does not compile, and the compiler is
right — two inserts genuinely race. But `insert` touches two things with totally
different access patterns: `data`, written once then read forever, and the
adjacency, read and written at scattered indices. One `&mut self` forces the
strictest rule over both. Splitting the struct along its real access pattern is
simultaneously what unblocks the borrow checker and what makes it fast: `data`
becomes a plain `&[f32]` (`Sync` for free, no locks, no atomics), and only the
adjacency needs `UnsafeCell` + 4096 sharded `RwLock`s. `n` locks would be 50 000
mutexes; one lock would serialise the build.

**The same mistake, smaller:** `fn select(&mut self, ...)` mutated nothing. That
one wrong word made `let v = self.vec_at(c.id)` fail to outlive the next call, so
the "fix" looked like `.to_vec()` — a heap allocation per candidate, ~32 per
insert. Narrowing the receiver to `&self` deleted them all. The borrow checker
was never the problem.

**`unsafe impl Sync` is a promise, so write down the terms.** The module header
of `src/par.rs` states the contract (`data` immutable after build starts; every
adjacency access under its shard lock; a node publishes its own edges before
anyone links back; no thread ever holds two locks, so no lock-order deadlock) and
`tests/parallel_equivalence.rs` is what enforces it. A race here never panics —
it silently drops edges and surfaces months later as "recall is a bit below the
paper". So the tests assert structure (zero orphans, degree in range) and a
recall floor, not just "it didn't crash".

**`Option::take` beats a raw-pointer borrow split.** Quantized search needs
`&q8` alive across calls taking `&mut self`. Casting to `*const Q8` works and I
wrote it that way first. Moving the field out with `take()`, running, and moving
it back compiles to the same code and makes the type system enforce what the
`unsafe` block could only assert in a comment.

**Determinism is a design choice, not a property.** Levels come from
`splitmix64(seed, id)` rather than a shared RNG — a shared RNG would be a second
contention point *and* would make a 4-thread build differ from a 1-thread build,
which would have made the thread sweep above uninterpretable. There is a test
that fails if anyone reintroduces one.

**`f32: !Ord`** because NaN, so `BinaryHeap` needs a newtype with a manual `Ord`,
and `Rev(Cand)` flips it into a min-heap. This is everywhere in real Rust.

---

## Layout

```
src/dist.rs   f32 L2 kernels: scalar / AVX2+FMA / AVX-512, runtime dispatch
              resolved once into a fn pointer at construction
src/q8.rs     int8 quantization + VNNI integer kernel + exactness tests
src/hnsw.rs   the graph: flat Vec<f32> corpus, flat Vec<u32> adjacency,
              epoch-stamped visited set, Alg.4 neighbour heuristic,
              f32 and int8 search paths
src/par.rs    concurrent builder: UnsafeCell + sharded RwLocks (the safety
              contract is the module header — read it before editing)
src/synth.rs  low-intrinsic-rank corpora, because "what data did you measure on"
              is the most load-bearing fact in any ANN benchmark
```

## Round 1 result worth keeping

ANN difficulty is governed by **intrinsic** dimension, not the ambient dimension
in your index config. Same 256-d index, varying the rank of the underlying
manifold:

| intrinsic rank | recall@10, ef=32 |
|---:|---:|
| 8 | 1.000 |
| 32 | 0.960 |
| 128 | 0.775 |

This is why published recall on SIFT/GloVe does not transfer to your corpus.
Measure the intrinsic dimension of your own embeddings before believing anyone's
numbers, including these.

The method that found it is the part worth copying: first sweep gave recall 0.66
and the tempting move was to tune `ef` and `m`. Instead — query the index with
vectors that are *in* the index. Self-retrieval recall@1 came back 1.000, which
proves the graph is correct and indicts the dataset. The benchmark was lying, not
the code.

## Next

- `-Z sanitizer=thread` over the build (needs nightly; the outcome tests are the
  stable-toolchain stand-in)
- product quantization — asymmetric distance via LUT, a completely different
  kernel shape again
- filtered search: the open problem in production vector DBs
