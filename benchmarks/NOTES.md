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

## 2026-09-17 — simdgroup tiles: all six configs clear 1.5x

| Config | Ours before tiles | Ours with tiles | Best valid public | Ratio |
|---|---|---|---|---|
| geo-trips 10M x 4 k256 | 0.0134 | (rows path) | FAISS 0.0682 | 5.1x |
| satellite 10M x 12 k32 | 0.0177 | (rows path) | scikit-learn 0.0883 | 5.0x |
| logs 10M x 32 k256 | 0.0785 | (rows path) | scikit-learn 0.3750 | 4.8x |
| single-cell 2M x 50 k64 | 0.0118 | (rows path) | scikit-learn 0.0564 | 4.8x |
| SIFT1M 1M x 128 k1024 | 0.0854 | 0.0562 | scikit-learn 0.1824 | 3.2x |
| GIST1M 1M x 960 k1024 | 0.6533 | 0.2898 | fast-pytorch-kmeans 0.5115 | 1.8x |

- **Metal simdgroup matrices are IEEE-quality**: dot-product error 1.9x eps at 960 dims (numpy/Accelerate: 1.85x),
  exact on integers. Apple's Neural Accelerator path (MetalPerformancePrimitives, used by MLX `@` and PyTorch MPS)
  is ~4x faster again but exact only on integers: ~12000x eps on fractional data, so it cannot be used where labels
  must be provably exact.
- **Distance kernel throughput at GIST shape:** direct (x-c)^2 pairs 1.6T terms/s; simdgroup tiles 5.1T; NAX matmul
  19.6T. float4 vectorisation and a dot-product form did NOT help the direct kernel (memory-bound, not compute-bound).
- **Triangle-inequality pruning is useless at 960 dims:** 78% of centers survive the |c - c_a| > 2 d_a test, so
  Elkan-style pruning cannot replace exact distances there.
- **Public libraries are label-accurate on real data:** scikit-learn, FAISS and fast-pytorch-kmeans each had 0 wrong
  labels vs float64 (float32 tolerance) on SIFT and GIST, despite the approximate matmul paths. The differentiator is
  that ours is provably exact, not that theirs is wrong in practice.
- **Multiple libraries in one process crash** (native, no traceback) when sklearn + faiss + torch are mixed; run each
  in its own process.

### Open: comparisons that still need to be made
1. **FAISS as people actually use it** — default `max_points_per_centroid=256` trains on a subsample (262k of 1M rows
   at k=1024). Disabled in bench.py for equal work, but the decision-relevant comparison is quality vs wall clock:
   time to reach a given inertia, ours on all data vs FAISS on its sample, both to convergence.
2. **ANN end-to-end** — build an IVF index from our centroids vs FAISS-trained centroids on SIFT1M (ground truth
   ships with the dataset) and compare recall@10 and search time at equal build cost.
3. **FAISS Metal** — `faiss.get_compile_options()` reports `MAC_METAL` in this 1.15.1 wheel; find out whether that
   exposes a GPU path on this machine before making any claim about FAISS here.

## 2026-09-17 — end-to-end on SIFT1M (bench_ann.py): quality per second and ANN recall

Per-iteration speed is not the decision-relevant number: FAISS trains an IVF quantizer on a 256-points-per-cluster
subsample (262k of 1M rows at k=1024) for 25 iterations, and scikit-learn's k-means++ init alone costs minutes.
Inertia is always measured by our exact pass over the full data, so the metric is identical for every method.

| Method | Train | Inertia (x best) | recall@10 nprobe 8 |
|---|---|---|---|
| ours, all rows, to convergence (36 iters) | 2.48 s | 1.0000 | 84.74% |
| ours, all rows, 25 iters (FAISS's budget) | 1.83 s | 1.0014 | 84.57% |
| ours, 262k subsample, to convergence | 1.06 s | 1.0093 | 84.26% |
| FAISS default (subsample, 25 iters) | 4.81 s | 1.0099 | 84.17% |
| FAISS all rows, 25 iters | 17.90 s | 1.0014 | 84.45% |
| scikit-learn all rows, 25 iters | 165.08 s | 1.0004 | 84.83% |

(after the fused k-means++ init, commit 5b55a50; the earlier run with 1.5 s init read 3.39 / 2.88 / 2.09 s for ours)

- **Same quality, 9.8x less time:** ours at 25 iterations and FAISS on all rows land on the same inertia (1.0014x
  best); ours takes 1.83 s, FAISS 17.90 s.
- **vs FAISS as configured by default:** 2.6x faster (1.83 s vs 4.81 s) with better inertia and +0.4 pt recall,
  while using all 1M rows instead of a 262k subsample.

- Ours gives the best clustering and the best recall in the least time: 1.7x faster than FAISS's default while using
  4x more data, 6.3x faster than FAISS on all rows, 69x faster than scikit-learn.
- IVF index build (0.75 s) and search times are the same for every centroid set, as expected - the centroids only
  move recall, and the spread is small (84.1-84.8% at nprobe 8).
- **k-means++ init was the hidden cost**: 18.1 s in NumPy at k=1024 (it dominated a 12 s "12 s total" run). On the GPU
  it is 1.5 s, and is now the largest single item in a 2.88 s run - the next thing to optimise.
- **FAISS has no GPU path on this machine**: `faiss.get_compile_options()` reports MAC_METAL and `get_num_gpus()`
  returns 1, but `faiss.Kmeans(gpu=True)` fails (GpuResourcesVector missing) and `index_cpu_to_gpu` returns a plain
  CPU Index (same search time as IndexFlatL2). FAISS here is CPU-only unless built from source.

## 2026-09-17 — shape sweep: where we win and where we don't

Ad-hoc synthetic configs via `bench.py --shape ROWS,DIMS,K` (recorded in results.jsonl, not part of SUITE). Ours is
the latest run; "best valid public" is the fastest library whose inertia matches ours within 1e-4.

| Shape | Ours s/pass | Best valid public | Ratio | Faster but worse quality |
|---|---|---|---|---|
| 100,000 x 8, k=8 | 0.0011 | faiss 0.0011 | 1.0x | — |
| 100,000 x 32, k=64 | 0.0019 | scikit-learn lloyd 0.0019 | 1.0x | — |
| 200,000 x 960, k=256 | 0.0458 | scikit-learn lloyd 0.1324 | 2.9x | fast-pytorch-kmeans 0.0296s (+3.3% inertia) |
| 1,000,000 x 4, k=8 | 0.0017 | fast-pytorch-kmeans 0.0041 | 2.5x | — |
| 1,000,000 x 32, k=64 | 0.0033 | scikit-learn lloyd 0.0143 | 4.3x | — |
| 1,000,000 x 32, k=4096 | 0.1305 | scikit-learn lloyd 0.3987 | 3.1x | — |
| 1,000,000 x 128, k=64 | 0.0094 | scikit-learn lloyd 0.0604 | 6.5x | — |
| 1,000,000 x 128, k=4096 | 0.2137 | scikit-learn lloyd 0.6778 | 3.2x | — |
| 1,000,000 x 256, k=1024 | 0.0892 | scikit-learn lloyd 0.3479 | 3.9x | — |
| 5,000,000 x 16, k=8192 | 0.4631 | scikit-learn lloyd 3.6235 | 7.8x | — |
| 10,000,000 x 8, k=8 | 0.0083 | fast-pytorch-kmeans 0.0577 | 7.0x | — |
| 10,000,000 x 64, k=1024 | 0.3786 | scikit-learn lloyd 1.2418 | 3.3x | — |

- **Below ~100k rows we only tie**: a pass costs 1-2 ms and GPU dispatch overhead dominates. Use scikit-learn there.
- **From 1M rows up we are 2.5-7.8x** across dims 4-960 and k 8-8192, at identical inertia.
- **Largest wins are at extreme k and at many rows with few dims**: 5M x 16 k=8192 is 7.8x (scikit-learn's elkan
  collapses there: 37 s/pass, 80x ours), 10M x 8 k=8 is 7.0x.
- **fast-pytorch-kmeans is faster than us in one corner** (200k x 960, k=256: 0.0296 s vs our 0.0458 s) but ends 3.3%
  worse because it zeroes empty clusters instead of relocating them; it is excluded as different work.
- **fast-pytorch-kmeans runs out of memory** at k>=1024 with many rows (dense k x rows mask: 16 GB at 1M x k=4096,
  164 GB at 5M x k=8192), so it is not an option for large-k work on this machine.
- **FAISS drifts from our inertia at large k and high dims** (+0.3% to +3.3%, and +17.7% at 1M x 128 k=64) because it
  splits large clusters to fill empty ones. scikit-learn, which relocates to far points like us, matches our inertia
  exactly on every shape in this sweep.

## 2026-09-18 — the "approximate to go faster" tools lose here

The standard answers to "full k-means is too slow" are FAISS's subsample default and scikit-learn's
MiniBatchKMeans. On this machine both are slower *and* worse than our exact run over all the data.

SIFT1M (1M x 128, k=1024), wall clock to a finished set of centroids, inertia measured by our exact pass:

| Method | Train | Inertia (x best) | recall@10 nprobe 8 |
|---|---|---|---|
| ours, all rows, to convergence | 2.09 s | 1.0000 | 84.89% |
| ours, all rows, 25 iters | 1.59 s | 1.0010 | 84.71% |
| FAISS default (262k subsample) | 5.02 s | 1.0102 | 84.17% |
| scikit-learn MiniBatchKMeans | 9.71 s | 1.0129 | 83.86% |

Synthetic 100M x 32, k=1024: ours 11.97 s to convergence (4 iterations, best inertia). **MiniBatchKMeans did not
finish in 84 minutes** at that shape (killed; ~700% CPU, 40 GB resident) - its per-batch assignment against k=1024
centres runs on the CPU for every batch, so the batching that saves memory does not save time here.

Takeaway for the write-up: on a machine where a full exact pass over 100M rows costs ~0.5 s, the approximations that
exist to protect a CPU budget stop paying for themselves - they cost both time and quality.

## 2026-09-18 — where the boundary actually is

Suite after the accumulation-path fix (ours vs the fastest public library matching our inertia):

| Config | Ours s/pass | Best public | Ratio |
|---|---|---|---|
| geo-trips 10M x 4 k256 | 0.0099 | FAISS 0.0661 | 6.7x |
| satellite 10M x 12 k32 | 0.0093 | scikit-learn 0.0711 | 7.7x |
| logs 10M x 32 k256 | 0.0700 | scikit-learn 0.2903 | 4.1x |
| single-cell 2M x 50 k64 | 0.0102 | scikit-learn 0.0289 | 2.8x |
| SIFT1M 1M x 128 k1024 | 0.0540 | fast-pytorch-kmeans 0.1746 | 3.2x |
| GIST1M 1M x 960 k1024 | 0.2899 | fast-pytorch-kmeans 0.6081 | 2.1x |

**Below ~100k rows the CPU libraries win**, and on a quiet machine it is not a tie: 100k x 32 k=64 scikit-learn
1.44 ms vs ours 2.82 ms (0.5x); 100k x 8 k=8 FAISS 0.69 ms vs ours 1.59 ms (0.4x). A pass that small costs 1-2 ms,
most of it GPU dispatch (~0.14 ms per kernel launch, several per pass), so no kernel tuning changes it - the fix
would be fusing the pass into one or two dispatches, which the deterministic accumulation design does not allow.
Recommendation in the README: use scikit-learn under ~100k rows; this library is for 1M rows and up.

Also fixed here: the deterministic path choice (sorted vs blocks) now switches on row count as well as k*(dims+2) -
geo-trips 10M x 4 k=256 went 0.0133 -> 0.0099 s/pass (25%), since the argsort cost more than the buffers it saved.

## 2026-09-18 — 1B-row regression after all the changes

`python3 -m mlx_kmeans --rows 1_000_000_000 --dims 8 --k 16`: 0.7 s to generate 32 GB on the GPU, 1.6 s to cluster
(3 iterations), 2.31 s wall. Final inertia 8.000078e9 against a theoretical optimum of 8.0e9 (1B rows x 8 dims x
unit variance) - the greedy k-means++ seeding lands in the right basin and converges immediately. The same run
before that change took 8 iterations and finished at 1.271e10, 59% worse.
