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

## 2026-09-17 — rounds 1–2 of the performance loop

- **MLX GPU matmul is not IEEE-float32-tight:** error vs float64 ~5,000x eps (Accelerate CPU: 2–10x eps). An
  expanded-distance hybrid with a provable candidate bound mislabeled up to 66k/100k points. Rejected.
- **Thread granularity, not branches, made the first kernel slow at large k*dims:** branch-free argmin inside
  4096-row block threads changed nothing; one GPU thread per row (branch-free argmin) was 9–43x faster.
- **Compensated (Neumaier) accumulation** on the GPU gives centers within ~1e-13 of float64. MLX Metal kernels do
  not reassociate floats (probe: Kahan error terms survive).
- **Rows vs pairs crossover:** per-(row, center) threads win from dims >= 64 (1.0–3.2x); rows path below.
- **First suite (partial, commit b5eeae0):** geo-trips 4.9x FAISS; satellite 5.0x scikit-learn; logs 3.8x
  fast-pytorch-kmeans; single-cell 1.35x fast-pytorch-kmeans (below target; scikit-learn/FAISS relocate empty
  clusters and reach 24% lower inertia there — ours keeps empty centers); SIFT1M 1.13x scikit-learn (below target);
  GIST1M ours 4.8 s vs scikit-learn 0.94 s (5x slower: per-pair kernel is memory-bound at 960 dims).
- **Incident:** the suite's fast-pytorch-kmeans GIST1M run wired 123 GB of GPU memory and kernel-panicked the machine
  (watchdog). Its safe mode test-allocates up to rows*dims*k*4 bytes; `iogpu.wired_limit_mb` had been raised to
  126000. bench.py now caps PyTorch MPS memory via watermark env vars (verified effective).
