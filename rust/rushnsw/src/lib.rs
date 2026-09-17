//! # rushnsw
//!
//! An HNSW approximate-nearest-neighbour index with hand-written AVX-512 /
//! AVX512-VNNI distance kernels, lock-sharded parallel construction, and int8
//! scalar quantization with exact rerank.
//!
//! ## Who wrote this
//!
//! Claude wrote this crate — the intrinsics, the concurrent builder, the
//! benchmarks, and the analysis of them. It is a learning artifact, built to be
//! read rather than depended on: no semver, no stability guarantee, and the API
//! changes whenever changing it teaches something.
//!
//! The benchmark figures in the README were measured on one 4-core container,
//! once. Every speedup claim ships with the control that isolates it, every
//! recall figure is computed against exact brute-force ground truth in the same
//! process, and the first measurement that turned out to be wrong is still in
//! the README next to the control that disproved it. Judge it on that. See
//! `DISCLAIMER.md` at the repository root.

pub mod dist;
pub mod hnsw;
pub mod par;
pub mod q8;
pub mod synth;
