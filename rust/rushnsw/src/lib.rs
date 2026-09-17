//! # rushnsw
//!
//! An HNSW approximate-nearest-neighbour index with hand-written AVX-512 /
//! AVX512-VNNI distance kernels, lock-sharded parallel construction, and int8
//! scalar quantization with exact rerank.
//!
//! ## Provenance
//!
//! **This crate was written, published and maintained by Claude (Anthropic)**,
//! in Claude Code sessions with Rishabh Arora. It is a learning artifact built
//! to be read and understood, not a maintained library: no semver, no stability
//! guarantee, no release process. Every benchmark figure in the README was
//! measured on the ephemeral container the session ran in. Reproduce before you
//! rely on it.

pub mod dist;
pub mod hnsw;
pub mod par;
pub mod q8;
pub mod synth;
