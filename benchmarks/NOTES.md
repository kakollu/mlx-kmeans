# Benchmark notes

Hand-written log of measurements and findings. Harness results live in `results.jsonl` / `../BENCHMARKS.md`;
this file records the earlier one-off measurements and the reasoning behind design choices.
Machine throughout: Apple M5 Max (18 CPU cores, 40 GPU cores), 128 GB unified memory.

## 2026-09-16/17 — from first version to fused kernel

Time for one Lloyd pass over 100M rows, 8 dims, k=16:

| Implementation | s/pass | Notes |
|---|---|---|
| NumPy, CPU (chunked, expanded distance) | 4.2 | |
| C, 1 core | 0.69 | |
| MLX built-in ops (first version) | 0.38 | ~7 GPU kernels per chunk; float32 expanded distance misassigned 1,019 of 4M points |
| C, 18 cores (GCD) | 0.049 | |
| MLX `metal_kernel` fused pass | 0.022 | distance + argmin + per-cluster sums in one read; exact vs float64 |
| Memory read floor (`mx.sum` over the 3.2 GB) | 0.006 | bandwidth ceiling for any pass |

1B rows, 8 dims, k=16, end to end:

| Version | Data gen | s/pass | Total |
|---|---|---|---|
| First MLX version (user's run) | ~30 s (NumPy RNG, single-threaded) | ~3.8 | ~40 s |
| kmeans.py, fused pass, `mx.random` data | 1.8 s | 0.215 | 3.75 s (8 passes) |
| kmeans.py, fused pass + Metal data-gen kernel | 0.7 s | 0.215 | 2.85 s (9 passes) |
| kmeans.mm (C++/Metal) | 0.65 s | 0.209 | 1.7 s (4 passes, different data) |
| kmeans_simple.py (plain MLX, expanded distance) | 1.5 s | 3.35 | 15.3 s (4 passes) |

Pass counts differ because each version generates different random data: over 20 seeds at 10M rows,
kmeans.py averaged 6.4 passes and kmeans_simple.py 5.9. Compare s/pass, not totals.

## Findings

- **The host language barely matters once the pass is one kernel.** C++/Metal vs Python/MLX `metal_kernel`: 3% per pass.
- **Submit one command buffer per 100M-row slice.** One buffer holding all 10 slices (32 GB) ran 0.225 s/pass vs 0.209 s.
  A Metal residency set did not change it.
- **C++ CPU loop:** a scalar `best` in the argmin (not `dist[bl]`) was 2x faster; accumulating into a per-job stack
  buffer instead of a shared array was another 2x (0.21 -> 0.049 s/pass at 100M).
- **Scaling (fused kernel, s/pass at 1B rows, extrapolated from 100M/10M):** d8/k16 0.22, k64 0.6, k256 4.3,
  k1024 63, k4096 ~1,300; d32/k16 1.2; d128/k16 18; d128/k256 ~1,300. Cost per row*k*dims rises ~20x above
  k*dims ~2,000. Cache tiling (rows x centers) only gave 1.4x at k=4096 and slowed small k: cause not found.
- **Memory limits:** GPU working set 123 GiB, max single buffer 80.6 GiB -> ~25B float32 values of data in practice.
- **Correctness bugs found:** uint32 kernel indices overflowed above 42 dims with 100M-row slices (fixed in
  kmeans.py by sizing slices; guarded in kmeans.mm). Per-block result buffers grew to GBs at large k*dims (fixed by
  growing the block size).
- **Local minima:** with k=16 hidden clusters, k-means++ on a sample rarely reached the ideal error (8/row) in 20
  seeds for either version; restarts (`n_init`) or greedy k-means++ would fix it.
