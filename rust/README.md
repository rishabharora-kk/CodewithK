# rust/

> Claude wrote these crates. Every speedup below ships with the baseline that
> isolates it. [More on that.](../DISCLAIMER.md)

| crate | what it is |
|---|---|
| [`rushnsw`](rushnsw) | Approximate nearest-neighbour index (HNSW) with SIMD distance kernels, parallel build, and int8 quantization. Two dependencies: `rayon`, and nothing else. |
