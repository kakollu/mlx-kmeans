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

## Small-data losses are latency, not arithmetic (September 20, 2026)

Measured on battery power, not the AC conditions the September 19 suite used. The mechanism and the
ratios are what matter here and both are robust to that; the absolute times are an upper bound.

Prompted by a 10k-row smoke test where scikit-learn finished a complete fit in 33 ms against our 95 ms. A single
case is noise; the question was whether a fixed per-call cost existed, since that compounds when a caller clusters
many small datasets in a loop. Two fixed costs were found, both round-trip latency rather than work.

**Seeding — fixed.** Init time was flat in row count and linear in k: at k=64 it cost about the same for 1k rows
as for 3M (16-26 ms), and cutting the sample tenfold at fixed k=64 moved it from 13.9 ms to 14.3 ms, i.e. not at
all. Against a measured bare MLX round trip of 142 us, the cost was one round trip per cluster. The `mx.eval` per
round was removed; MLX then pipelines the rounds. Seeding is 2-4x faster (k=256 on 1M rows: 60 -> 14 ms; k=1024 on
GIST-shaped data: 0.49 -> 0.26 s), for a bounded ~90 MB of extra working memory that does not grow with k. The
saving is a fixed few tens of ms per fit, so complete fits measured 1.1-1.9x faster between 1k and 100k rows and
the gain fades to a few percent at millions of rows - at SIFT1M's ~2.5 s training the 0.23 s saved is ~1.1x. Centers are bit-identical across 5 shapes x 3 seeds at k=8..1024, since only the timing
of evaluation changed.

Two alternatives were measured and rejected: running the rounds in NumPy on the host wins only below a few
thousand sample rows and loses badly above (k=64 on 100k rows: 217 ms against 15.8 ms), and it changes the RNG
stream, so it is not a drop-in. Keeping a periodic sync to bound memory proved unnecessary, as peak MLX memory
was the same at k=256 and k=1024.

**Per Lloyd pass — still present.** A pass has a ~0.9 ms floor independent of row count: it crosses to the host
for the float64 center update, the empty-cluster check and the convergence test. At 32 dims, k=64, a pass costs
0.98 ms at 1k rows and 1.03 ms at 100k; work only overtakes the floor above roughly 300k rows, growing at ~2 ns
per row thereafter. This is why the recorded 100k x 32, k=64 per-pass loss reproduces after the seeding fix
(2.2 ms against scikit-learn's 1.2 ms) - seeding is not part of a per-pass number. Removing this floor means
keeping the center update and convergence test on the GPU, which would move float64 accumulation and
scikit-learn-exact empty-cluster relocation off the host; not attempted, as correctness ranks above speed here.

## Re-testing the tuning after the GPU-teaching experiments (September 23, 2026, battery)

Writing `examples/learn_gpu_kernels.py` produced three hypotheses about the production kernels. All three
were measured on the M5 Max. One was already shipped; the other two were wrong, and the existing constants
are correct. Times are on battery, so the ratios are the quotable part.

**Confirmed and shipped: lazy k-means++ seeding.** A/B against the pre-fix implementation taken from git
(ed9eb4b), not from memory: 10k x 32 k=64, 17.6 -> 4.0 ms (4.40x); 1M x 32 k=256, 60.2 -> 13.7 ms (4.39x);
1M x 128 k=1024, 247.5 -> 66.6 ms (3.71x). Centres bit-identical in every case.

**Rejected: column-major (SoA) input layout.** The standard coalescing argument says X[i*D+j] is the wrong
layout. Measured on an 8M-row distance kernel: 128 dims gives a consistent ~18%, but 32 dims gives 0.92-0.99
and 6 dims moves between 0.76 and 1.00 across runs, i.e. inside noise. A 24-byte row means one 128-byte cache
line already serves several neighbouring threads. The `rows` path only runs below 64 dims, where the effect is
noise, and the high-dimensional path is `tiles`, which already transposes the centres. Converting would also
cost a full transpose of the input. Not worth it.

**Rejected: more accumulation blocks.** ACC_MAX_BLOCKS caps `blocks` at 4096 threads, which at 10M rows is
2442 rows per thread - the very pattern that cost 9-43x in the assignment kernel. Raising it makes things
much worse, because the per-block buffers must be zero-filled and then reduced:

| blocks | rows/thread | 10M x 6 k=8 | 10M x 12 k=32 |
|---:|---:|---:|---:|
| 4,096 (shipped) | 2,442 | 3.80 ms | 6.83 ms |
| 16,384 | 611 | 5.89 ms (0.65x) | 13.75 ms (0.50x) |
| 65,536 | 153 | 23.98 ms (0.16x) | 28.64 ms (0.24x) |
| 262,144 | 39 | 92.13 ms (0.04x) | 80.00 ms (0.09x) |

The existing value is optimal. Note also that block count changes how the Kahan compensation splits, so
results are not bit-identical across block counts - another reason not to make it adaptive casually.

**Where the remaining headroom is, and why it is not free.** Against a measured 540 GB/s streaming ceiling:
the satellite-shaped assignment pass runs at 38% of it and accumulation at 16%, so neither is bandwidth-bound.
The gap is the compensated summation and the scattered per-thread writes, i.e. the price of determinism.
Measured directly, the opt-in `atomic` path is 2.2-4.3x faster than the deterministic default
(10M x 6 k=8: 3.87 -> 1.79 ms; 10M x 12 k=32: 6.93 -> 1.60 ms; 2M x 50 k=64: 9.53 -> 3.68 ms), consistent
with the 1.7-4.5x already documented. The SIFT-shaped `tiles` assignment sits at 2% of the bandwidth ceiling
because it is compute-bound, which is the correct regime for it.

## FlashAttention-style fusion of the tiles path: analysis (September 24, 2026, battery)

FlashAttention's central trick is not tiling for its own sake - it is never materialising the large
intermediate matrix, keeping blocks on-chip and carrying running statistics instead. Our `tiles` assignment
path does materialise one: `_DOT_TILES_SRC` writes an (m x k) dot matrix to device memory and `_MARGIN_SRC`
reads it straight back. So the technique applies here directly. Measured, per chunk:

| Workload | dot kernel (writes m x k) | margin kernel (reads it) | intermediate traffic/pass | vs input X |
|---|---:|---:|---:|---:|
| SIFT1M 1M x 128, k=1024 | 3.63 ms, 537 MB | 3.10 ms, 537 MB | 8.6 GB | **16.8x** |
| GIST 1M x 960, k=1024 | 25.11 ms, 537 MB | 5.75 ms, 537 MB | 8.6 GB | 2.2x |

At SIFT-like shapes the intermediate is 16.8x the actual data. GIST is less affected because its matmul is
genuinely expensive, so the write is a smaller share.

**The cheap version does not work.** Shrinking the chunk so the dot matrix becomes cache-resident is a
one-constant change, and it is monotonically worse: at SIFT, 32768 rows gives 0.86x, 8192 gives 0.70x and
2048 gives 0.28x (labels identical throughout). The cause is the per-chunk launch and sync: at chunk 2048
there are 488 launch/sync pairs, and at the measured ~0.14 ms round trip that is ~137 ms of the 178 ms.
Tiling alone cannot help while the two kernels remain separate - which is the argument for real fusion,
since one kernel removes the intermediate and the per-chunk syncs together.

**A fused pass can stay provably exact.** The obstacle is that the pruning bound needs the global minimum,
which a streaming kernel does not have. The resolution is the same one FlashAttention uses for the softmax
maximum: carry a running statistic. Test each centre against the *running* bound, which is always looser than
the final bound, and compute its exact distance immediately. The set examined is therefore a superset of the
true candidate set, so the nearest centre is always evaluated exactly. Measured cost of that looseness:

| Workload | survivors/row, final bound | survivors/row, running bound | share of k |
|---|---:|---:|---:|
| 128d, k=1024 | 1.00 | 6.7 | 0.65% |
| 960d, k=1024 | 1.07 | - | 0.10% |
| 64d, k=256 | 1.00 | - | 0.39% |

6.7 exact distances per row against a matmul of 1024 centres is roughly 0.6% extra arithmetic.

**Expected payoff.** The dot kernel's 537 MB write is ~1.07 ms of its 3.63 ms at ~500 GB/s, so the matmul
itself is ~2.5 ms. A fused kernel should cost about that plus the ~0.6% extra, against 6.73 ms for the
current pair - roughly 2.5x on the assignment step at SIFT-like shapes, and less at GIST-like ones where
the matmul dominates anyway. Not attempted yet: it needs a new simdgroup kernel that keeps 8x8 tiles in
threadgroup memory, carries a per-row running minimum across centre blocks, and recomputes exact distances
inline. Correctness must be checked against the existing path before it could replace it.

### Implemented and rejected: the fused kernel is correct but slower (September 24, 2026, battery)

Three variants were written and validated against `_tiles_nearest`. All produce **bit-identical labels and
bit-identical distances** on every shape tested, confirming the running-bound exactness argument above holds
in practice. None is faster.

| Variant | 1M x 128 k=1024 | 200k x 960 k=1024 | 200k x 64 k=256 |
|---|---:|---:|---:|
| v1, one simdgroup streams all centres | 0.93x | 0.41x | 0.87x |
| v2, centre slice per simdgroup (SG=4/8/16) | 0.79 / 0.65 / 0.50x | 0.43 / 0.39 / 0.33x | 0.61 / 0.48 / 0.39x |
| v3, B centre-blocks batched per scan (B=4..32) | **0.97** / 0.95 / 0.90 / 0.73x | 0.42x | 0.86x |

The estimated 2.5x did not appear, and the reason is utilisation rather than bandwidth. The shipped pair
lets each kernel run at its own peak: `_DOT_TILES_SRC` is a dense matmul with no divergence and no barriers,
and `_MARGIN_SRC` is a simple streaming scan. Fusing them interleaves the two, so 24 of 32 lanes idle behind
a barrier during every scan phase, and the matmul never reaches the throughput it has on its own. That loss
exceeds the 2.1 ms of intermediate traffic the fusion saves per chunk.

v2 is the clearest evidence: adding simdgroups made it monotonically worse, so the limit was never a shortage
of parallel work. v3 recovers most of the gap by batching (0.97x at SIFT with B=4) but does not cross 1.0x.

Not shipped: a correct-but-slower path behind a flag is maintenance cost with no benefit. What remains
untried is a scan that keeps all 32 lanes busy - one lane per (row, centre) entry of the 8x8 tile, with the
per-row running minimum reduced through simd shuffles instead of eight lanes scanning serially. That removes
the idle-lane problem the measurements point at, and is the form a future attempt should take.

## Looking for redundant work between iterations (September 24, 2026, AC power)

Three candidates for precomputing what a pass redoes. One is worth building, one is not, one is marginal.

**|x|^2 caching: measured, rejected.** `_MARGIN_SRC` recomputes |x|^2 from X on every pass although it never
changes while the centers move. Handing it in precomputed made the margin kernel 1.06-1.17x faster in isolation,
which did not survive end to end: over 10 Lloyd iterations it measured 1.02x, 1.02x, 0.99x and 0.97x on four
tiles-path shapes. The margin kernel already reads X for the exact recompute, so X is hot in cache and the |x|^2
loop is nearly free. The plumbing (an optional per-fit cache threaded through lloyd_step, _gpu_pass and _nearest)
was written, verified bit-identical, measured and then reverted rather than carried as unused complexity with a
cache-invalidation surface.

**Bound-based pruning (Hamerly/Elkan): real but much smaller than it first looks.** Points that keep their
cluster need no distances at all, and assignments settle quickly - measured churn per iteration falls to 3-6% by
iteration five (500k x 128 k=256: 37.1, 12.8, 9.2, 7.7, 6.6, 5.7, 5.1, 4.5, 4.1%). That number is misleading.
What matters is not how many points keep their cluster but how many can be *proven* to keep it, and a faithful
float64 Hamerly simulation gives a far weaker result - the share still needing a full k-distance scan, averaged
over ten iterations:

| Shape | still needs a full scan | implied ceiling |
|---|---:|---:|
| 4 dims, k=32 | 48.3% | ~2.1x |
| 12 dims, k=32 | 61.2% | ~1.6x |
| 48 dims, k=64 | 81.4% | ~1.2x |
| 128 dims, k=256 | 89.3% | ~1.1x |
| 960 dims, k=64 | 89.0% | ~1.1x |

Effectiveness collapses with dimension, which agrees with the existing finding that triangle-inequality pruning
is useless at 960 dims. Two further costs specific to this implementation: the work becomes irregular, and with
threads in lockstep one point needing a full scan makes its whole simdgroup pay unless the survivors are
compacted first; and it adds two floats of state per row (8 GB at a billion rows). So the honest prospect is
perhaps 1.3-1.6x on 4-12 dimensional data after compaction, in exactly the regime where this library is already
strongest, and nothing at high dimensions. Worth building for the low-dimensional case; not a general answer.

**Transposing the centers on the host: marginal.** `_tiles_nearest` does `mx.array(np.ascontiguousarray(
np.array(Cm).T))` every pass, which is a GPU-to-host round trip. On GPU via `mx.contiguous(Cm.T)` it is 5.6x
faster at k=1024, d=960 (0.933 -> 0.168 ms) and identical, but *slower* at d=128 (0.090 -> 0.147 ms), and even
the 0.933 ms is 0.4% of a 233 ms GIST pass. Not worth a size-dependent branch on these numbers.

## Things that did not work, and the limits that remain (September 24, 2026, AC power)

Four algorithm-level attempts, all measured, none adopted. Recorded because each is a natural idea and the
next person will have it too.

**FlashAttention-style fusion of the tiles path.** The path materialises an (m x k) dot matrix and reads it
straight back - 8.6 GB of intermediate traffic per pass at SIFT shape against 0.51 GB of input. Three fused
variants were written and all produce bit-identical labels and distances, confirming the running-bound
exactness argument. None is faster: 0.97x at best (batched centre blocks), 0.41x at GIST shape. Fusing
interleaves a dense matmul with a scan that idles 24 of 32 lanes, and that utilisation loss exceeds the
2.1 ms per chunk the fusion saves. Adding simdgroups made it monotonically worse, so parallelism was never
the limit. Untried: a scan keeping all 32 lanes busy, reducing the per-row minimum through simd shuffles.

**Deterministic on-chip accumulation in fixed-point integers.** Float threadgroup atomics are 2.5-3.5x
faster than the compensated device-memory path at small k*(dims+2), but they are not reproducible: five fits
from the same data and seed gave five different answers on every shape tested (inertia spread 1e-10 to
9e-07). Integer addition is exact and order-independent, so integer atomics are deterministic by
construction - and that part worked, giving identical results across runs. It fails on range. 64-bit
threadgroup atomics are not supported on this hardware (`atomic_ulong` aborts the process), and 32 bits must
be split between the value and headroom for the block size, so precision and speed trade directly against
each other:

| block size | bits for value | time | vs default | accuracy vs float64 |
|---:|---:|---:|---:|---:|
| 4096 | 19 | 2.20 ms | 2.62x | 7.7e-06 |
| 1024 | 21 | 3.18 ms | 1.82x | 6.5e-06 |
| 256 | 23 | 13.33 ms | 0.43x | 6.3e-07 |
| 64 | 25 | 55.52 ms | 0.10x | 5.1e-07 |

Where it is fast it is inaccurate; where it is accurate it is slow, and even the accurate end is 10x worse
than the float atomics it would replace and six orders worse than the compensated default. With 64-bit
atomics there would be ~40 bits for the value, giving Kahan-level accuracy at on-chip speed; that is the
design that would work and this machine cannot run it.

**Bound-based pruning: Hamerly and Yinyang.** Assignments settle quickly - churn falls to 3-6% by iteration
five - but what matters is how many points can be *proven* unchanged. Hamerly proves few, and fewer as
dimension rises (11% at 4 dims, 52% at 12, 89% at 128 and 960). Yinyang is much better because it keeps a
lower bound per group of centres rather than one overall: 26.2% of the n*k distances survive at 4 dims k=256,
49.3% at 128 dims k=1024, and it keeps improving with iteration (15% by the eighth). That is a real
reduction in work, and it is exact.

It still loses on this machine, for opposite reasons at the two ends:

| config | distance work | bound I/O | survives | net |
|---|---:|---:|---:|---:|
| geo 10M x 4, k=256 | 2.8 ms | 4.5 ms | 26% | -2.4 ms |
| satellite 10M x 12, k=32 | 1.0 ms | 0.7 ms | 75% | -0.4 ms |
| logs 10M x 32, k=256 | 22.3 ms | 4.5 ms | 64% | **+3.5 ms** |
| SIFT 1M x 128, k=1024 | 18.3 ms | 1.8 ms | 49% | -5.0 ms |
| GIST 1M x 960, k=1024 | 137.5 ms | 1.8 ms | 49% | -25.8 ms |

At low dimensions the n*(k/10) lower bounds cost more to read and write than the distances they remove -
pruning only pays when a distance is expensive per point, and at 4 dims it is nearly free. At high
dimensions the pruned work is irregular and cannot use the simdgroup matmul, dropping from 14.3 to about
6 TFLOP/s, so Yinyang must prune more than 58% merely to break even and it prunes 51%. Only `logs` nets
positive, and by 1.2x on the pass. The general lesson is that the hardware has moved the crossover: doing
more arithmetic at the matmul rate beats doing less of it irregularly, by a factor of 2.4x here.

## Where the assignment kernels actually stand (September 24, 2026, AC power)

Going after a 2x on the `logs` shape (10M x 32, k=256), where assignment is most of the pass. Four ideas
measured before the one that worked, all on 4M x 32, k=256 against the 10.9 ms scalar rows kernel.

| Idea | Result | Why |
|---|---:|---|
| float4 loads of the centre table | 1.01x | not load-issue bound, as assumed - the compiler already vectorises |
| 2 rows per thread (reuse each centre load) | 0.81x | 64 floats per thread spills; helps only at 16 dims (1.10x) |
| 4 and 8 rows per thread | 0.10x | spills badly |
| expanded form, one FMA per dim | 1.26x | half the instructions, but then bound by the centre loads |

The expanded-form number is the useful one, because it says the scalar path is nearly finished: the direct
form does 98.3 GFLOP in 10.9 ms, which is **9.0 of about 11 TFLOP/s of scalar peak**. Halving the
instruction count buys 1.26x, not 2x, because the kernel stops being ALU-bound and starts waiting on
memory. There is no 2x left in a scalar kernel at this shape - only the matrix units are 4x faster.

Two further things that cost a lot and would not be guessed from reading the code:

**Arrays of simdgroup matrices spill when indexed in two dimensions.** Rewriting `a0[DK], a1[DK]` as
`a[RK][DK]` - the same 16 matrices - turned a 16.0 ms kernel into 65.1 ms at 64 dims. Flat arrays with a
single loop index are fine; two indices are not. This invalidated a whole table of measurements before it
was noticed, and the only reason it was noticed is that the numbers disagreed with a previous run.

**A candidate array indexed by a running count costs more than the multiply it protects.** `uint cand[M];
... cand[nc] = c; nc++` is the obvious way to collect centres inside the error window, and it is what the
margin kernel does. Because `nc` is a runtime value the array cannot live in registers, so it goes to
scratch memory: 11.3 ms against 5.7 ms for the same kernel keeping a fixed number of candidates in named
registers, on a multiply that costs 5.3 ms. In the margin kernel this is hidden because that kernel is
memory-bound on the distance matrix it reads; in a fused kernel there is nothing to hide behind.

### What the one-pass path does not cover

It is exact wherever it runs, but it only runs on 16-96 dims with k a multiple of 16, which is one of the
six suite configs (`logs`, 1.29x). The rest are out of range and each needs different work:

| Config | Why it is excluded | What it would take |
|---|---|---|
| geo-trips 10M x 4 | 4 dims | zero-pad to 8 - but 8 dims measured 0.88x, so probably not worth it |
| satellite 10M x 12 | 12 dims | pad to 16, but at k=32 that shape measured 0.38x |
| single-cell 2M x 50 | 50 dims | pad to 56; d=48/k=64 measured 1.60x, so this one should pay |
| sift 1M x 128 | above 96 dims | stage the row tiles in threadgroup memory instead of registers |
| gist 1M x 960 | above 96 dims | not the bottleneck: at 960 dims the pass is 138 ms of multiply and only 18 ms of distance-matrix traffic, so fusing saves 8%. It is not GEMM efficiency either - see the section below, which measures the ceiling and finds this kernel already at 94% of it |

Padding is exact rather than approximate: trailing zero dimensions add `+0.0f` terms at the end of each
sum, which cannot change a float32 accumulator, so labels and distances stay bit-identical.

## Accumulation: where its bandwidth was going (September 24, 2026, AC power)

After the assignment kernels got faster, accumulation was 40-70% of a Lloyd step and ran at 5-9x the cost
of one pass over the data. Decomposing it on 2M x 50, k=64 (floor 0.89 ms, one read of X):

| Kernel | Time | Notes |
|---|---:|---|
| read only, no scatter | 1.97 ms | a dependent chain of adds into one register - slower than scattering |
| threadgroup scatter, non-atomic | 1.57 ms | **wrong results**, priced only to isolate the scatter: 1.8x floor |
| threadgroup scatter, atomic (shipped) | 2.88 ms | atomicity costs 1.8x on top |
| sorted (the default) | 4.55 ms | |

So the scatter was never the problem. Two things were.

**Atomics were only needed because of how lanes were assigned.** The on-chip kernel gives each lane a row,
so 32 lanes race for one cluster slot. Give each lane a *dimension* and the 32 lanes of a row write 32
different addresses - no conflict, no atomics, rows consumed in index order, and the Kahan carry can sit on
chip beside each entry because every entry now has exactly one writer. Throughput is identical either way.

**The sorted path kept its running totals in device memory.** A segment belongs to one cluster, so its
accumulator is dims+2 floats, not k*(dims+2) - small enough to live in a simdgroup's registers. It was
instead read-modify-writing dims+2 floats in device memory per row, in the sums and again in the carries:
four times the traffic of the data. Fixing it was worth 5.6x on GIST1M, landing at 1.1x the memory floor.

Rejected along the way:

- **Plain float32 in the tile**, no compensation: 1.6 ms against 3.2 ms, but 7.6e-8 against float64 where
  the compensated version is 1.1e-16. Not a default.
- **Several simdgroups per threadgroup, each with its own tile**, to raise occupancy: slower (1.39 ms
  against 1.08 ms). N private tiles cost a core exactly what N threadgroups do, so there is no gain; only a
  *shared* tile would raise occupancy, and that brings the conflicts back.
- **One row at a time per simdgroup without lookahead**: 4.8 ms. The store address depends on a load of
  `labels[i]`, so the simdgroup has one row of memory parallelism. Loading 8 rows before adding any of them
  - the adds still sequential and in order - takes it to 1.6 ms.
- **Tuning rows-per-block**: the wrong knob. What matters is the block *count* - ~500 was best at 500k, 2M
  and 10M rows alike - because every extra block re-zeroes and writes out another tile. Fixing rows per
  block instead cost up to 2.4x at 10M rows.

### The register-candidate trick does not transfer to the margin kernel

Replacing `uint cand[M]` with candidates in named registers was worth 2x in the fused assignment kernel, so
the same change was tried in the margin kernel, which collects candidates the same way. It **lost**: GIST1M
went from 920 ms to 1010 ms for the suite config, and the suite total from 1319 ms to 1406 ms. Reverted.

Two reasons, both absent in the fused kernel. The margin kernel is memory-bound streaming a k-wide row of
dot products, so the scratch-memory traffic the array costs is hidden behind that read, while the extra ALU
of a four-deep insertion network is not. And keeping only the R smallest means discarded centres cannot be
tested individually, so the escape test has to use the largest |c|^2 over all centres - conservative enough
at 960 dims to send far more rows down the full-rescan path than the array version's per-centre test did.

The lesson is narrower than "registers beat arrays": it holds when the kernel is compute-bound and the
array is the only thing touching memory, and reverses when the kernel is already streaming.

## The tile matmul is at 94% of the hardware ceiling (September 24, 2026, AC power)

With assignment and accumulation both improved, the suite was 93-94% assignment on the two high-dimensional
configs, and 81% of *that* was the multiply: GIST1M spends 131.6 ms of a 175 ms Lloyd step in
`kmeans_dot_tiles16`, at 14.9 TFLOP/s. This repository had "about 40 TFLOP/s" on record as the machine's
peak, which made that look like a 2.7x inefficiency and kept it on the list of things to fix. **That number
was wrong, and the search it motivated was chasing nothing.**

Two measurements settle it.

**The multiply is not waiting on memory.** Holding the tile loads exactly constant and repeating each
accumulate into the same accumulator - which cannot be optimised away, since each depends on the last -
scales the arithmetic with the traffic fixed:

| MACs per load | Time (65536 x 960, k=1024) | vs 1x | Issued |
|---:|---:|---:|---:|
| 1x | 8.77 ms | 1.00x | 14.7 TFLOP/s |
| 2x | 16.21 ms | 1.85x | 15.9 TFLOP/s |
| 4x | 31.89 ms | 3.64x | 16.2 TFLOP/s |

Time tracks the multiply count almost exactly, and the issued rate is flat at ~16. Hoisting the X tiles out
of the loop - halving the tile loads, same multiplies - buys 6% at 960 dims.

**And ~16 TFLOP/s is all the instruction does.** A loop of `simdgroup_multiply_accumulate` with *no loads in
it at all*, sweeping the number of independent accumulators so the pipeline is full, and the simdgroup count
and threadgroup size so the GPU is full:

| Accumulators | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| TFLOP/s | 3.0 | 7.4 | 9.1 | 14.6 | 15.2 | 15.2 |

It saturates at **15.7 TFLOP/s** (best over threadgroup sizes and simdgroup counts). Eight independent
accumulators are needed to get there, which the 2x2-blocked kernel already has via its four accumulators
plus the loop's own overlap. So `kmeans_dot_tiles16` at 14.7 is at **94% of the ceiling**, and the earlier
finding that 4x4 register blocking bought only 15.1 against 14.6 was not a failure to optimise - it was the
ceiling, seen from below.

MLX's `@` does reach 28.7 TFLOP/s at 960 dims, so the *chip* can go faster - through different hardware, not
through this instruction. It is still not usable: its error is 5909x eps against this kernel's 9.4x, and a
candidate window widened by 630x admits every one of the 1024 centres, which is exactly what the earlier
NAX experiment measured (0.02-0.08x). Emulating float32 from it costs three of those matmuls, which at
3/28.7 against 1/14.7 is slower than just doing the work correctly.

**What this closes.** There is no large win left in exact assignment at high dimensions on this machine. The
remaining 19% of GIST's assignment is the margin kernel streaming the dot matrix, and sizing that matrix to
stay in cache does not help either - the chunk sweep below is monotonic the wrong way, because smaller
chunks cost more in dispatch and synchronisation than they save in traffic:

| Dot chunk | GIST 1M x 960 | SIFT 1M x 128 | logs 10M x 32 |
|---|---:|---:|---:|
| 512 MB (current) | 161.9 ms | 33.3 ms | 55.5 ms |
| 128 MB | 173.5 ms | 40.1 ms | 65.7 ms |
| 32 MB | 207.7 ms | 57.4 ms | 85.6 ms |
| 8 MB | 361.1 ms | 160.9 ms | 245.8 ms |

### What is left in the high-dimensional path, and what it would cost

With the multiply at its ceiling, GIST1M's 175 ms Lloyd step is 131.6 ms of multiply that cannot be
improved, 9.3 ms of accumulation already at 1.1x its memory floor, and ~31 ms of margin kernel. So the whole
step is **within about 24% of the floor implied by the hardware and the exactness guarantee**.

The one measured opportunity inside that 31 ms: the margin kernel recomputes |x|^2 for every row on every
pass, which at 960 dims is a second full read of X. Supplying it precomputed instead (labels bit-identical):

| Config | margin now | with |x|^2 supplied | saved per pass |
|---|---:|---:|---:|
| GIST1M 1M x 960, k=1024 | 31.2 ms | 23.5 ms | 7.7 ms (4% of the step) |
| SIFT1M 1M x 128, k=1024 | 20.9 ms | 14.2 ms | 6.7 ms (19% of the step) |

Computing it per pass buys nothing - the MLX reduction costs the 8 ms it saves. The gain only exists if it
is computed **once per fit**, since X does not change, and that is the catch: caching it against the data
would be a correctness risk, not just a performance one, because a stale |x|^2 makes the error bound wrong
and can change labels rather than merely slowing things down. It has to be threaded explicitly from the
estimator, which owns the arrays for the life of a fit, rather than cached by identity.

## Reconciling "94% of the ceiling" with the shrinking lead at large k and d

The benchmark lead falls monotonically with dimension - 9.4x at 12 dims, 7.0x at 32, 4.5x at 128, 2.8x at
960 - which looks like it contradicts the ceiling result above. It does not, and the difference matters for
what to build next.

Measured against what an approximate implementation on this machine can do (`mx.matmul` in the expanded
form, argmin taken on trust, no verification), 500k rows and k=1024:

| dims | ours, exact | approximate | ratio | our TFLOP/s | its TFLOP/s |
|---:|---:|---:|---:|---:|---:|
| 128 | 16.9 ms | 28.6 ms | **0.59x** | 7.8 | 4.6 |
| 256 | 26.6 ms | 29.7 ms | 0.90x | 9.8 | 8.8 |
| 384 | 36.5 ms | 33.4 ms | 1.09x | 10.8 | 11.8 |
| 512 | 47.2 ms | 34.6 ms | 1.37x | 11.1 | 15.2 |
| 960 | 81.8 ms | 46.0 ms | **1.78x** | 12.0 | 21.4 |

Three things follow.

**Our absolute efficiency rises with dimension, it does not fall** - 7.8 to 12.0 TFLOP/s. The shrinking lead
over scikit-learn, FAISS and fast-pytorch-kmeans is mostly those libraries getting *better*: at low
dimensions their per-row overheads dominate and we look 9x faster, while at 960 dims the problem is a pure
GEMM where their BLAS and MPS paths are at least respectable.

**Below about 320 dims, exactness is free.** We beat even the unverified approximate path, because it has to
materialise an n x k matrix and reduce it while our kernels do not. There is nothing to trade away here.

**Above about 320 dims there is a real gap, and it is precision, not verification.** At 960 dims ours is
162.2 ms (131.6 multiply + 30.5 margin) against 91.2 ms (68.6 multiply + ~22 argmin). Even with a *free*
verification we would still be 1.44x behind, because the multiply alone is 1.92x: 14.9 TFLOP/s through
`simdgroup_multiply_accumulate`, whose ceiling is 15.7, against 28.7 through the hardware MLX reaches. The
tax is being obliged to compute the dot product accurately enough for the error bound to be usable, and it
cannot be recovered by tuning - only by choosing a different guarantee. (Emulating float32 from the fast
path needs three of its matmuls, 3/28.7 against 1/14.7, which is slower.)

At 960 dims and k=1024 the approximate path disagreed with exact labels on 4,038 rows out of 1,000,000
(0.4%). Whether that matters is a question about the application - for IVF training it plausibly does not -
which is the argument for an opt-in approximate mode confined to high dimensions, and the argument against
making it the default anywhere.

## The approximate path, measured (September 24, 2026, AC power)

Following the reconciliation above, `KMeans(exact=False)` exists so the trade can be measured rather than
argued about. It takes the argmin of matmul-derived distances on trust - what a k-means written directly on
MLX does - and then computes the distance to the chosen centre exactly, so `inertia_` is the true inertia of
the labels returned and an approximate run stays comparable with an exact one.

It is ignored below 384 dims, because there the exact kernels are faster as well as exact. One assignment
pass from identical centres, k=1024:

| Shape | exact | approx | speedup | labels differing | extra inertia |
|---|---:|---:|---:|---:|---:|
| 500k x 128 | 16.7 ms | 29.3 ms | 0.57x *(flag ignored)* | 973 / 500k | 3.2e-07 |
| 500k x 384 | 36.5 ms | 35.4 ms | 1.03x | 790 / 500k | 1.2e-07 |
| 500k x 768 | 66.7 ms | 43.7 ms | 1.53x | 726 / 500k | 6.0e-08 |
| 200k x 960 | 33.5 ms | 20.5 ms | 1.63x | 309 / 200k | 6.1e-08 |

Disagreement is ~0.15% of rows and each one costs about 1e-7 of the distance - the approximate choice is
almost always a near-tie. Measuring it at identical centres matters: comparing labels after two full fits
conflates the approximation with the two runs reaching different local minima, which is a much larger effect
(12.8% of labels, on a case where the approximate fit ended with *lower* inertia).

**On the use case it exists for, it costs nothing measurable.** GIST1M, 1M x 960, k=1024, 12 iterations,
both sets of centroids scored by the same exact routine and then used to build a FAISS IVF index:

| | fit | inertia | recall@10, nprobe 1 | nprobe 8 | nprobe 32 |
|---|---:|---:|---:|---:|---:|
| exact | 2.84 s | — | 25.42% | 68.07% | 92.44% |
| approx | 1.77 s | +1.86e-04 | 25.57% | 68.49% | 92.49% |

1.61x faster for 0.019% more inertia, and recall the same to within noise - the approximate centroids score
marginally *higher* at every nprobe, which is a different local minimum rather than an improvement and
should not be read as one. Reproduce with `scripts/compare_exact_approx.py --ann`.

The honest summary for a user: above 384 dims this is a real speedup and the clustering is very slightly
worse in a way that did not reach the downstream metric; below 384 dims there is nothing to trade, because
exact is already faster.

## Four things that should have worked and did not (September 24, 2026, AC power)

After the cascade and the cache-reload path landed, the next four ideas all measured out too small to keep.
Recorded because each is the obvious next thing to try.

**float16 for the approximate matmul.** This machine does fp16 matmul at 52 TFLOP/s against 36.2 for fp32,
and the quality holds: on GIST1M the fitted inertia was *better* than the fp32 approximate path
(+1.3e-05 against +1.9e-04) and recall@10 was 92.36% against 92.44% exact. But the fit only went 0.99s to
0.93s - **1.06x** - because assignment is already a minority of an approximate fit. Not worth a new failure
mode for 6%, and the failure mode is sharp: SIFT's |x|^2 is 2.6e5 against float16's 6.55e4 ceiling, so
every row overflows and 499,409 of 500,000 labels come back wrong. A Cauchy-Schwarz guard
(sqrt(max|x|^2) * sqrt(max|c|^2) < 65504/4) predicts both cases exactly, and would be the right guard if
this were ever worth having.

**Multi-row lanes for small dimensions.** The accumulator gives each lane one dimension, so at 4 dims it
uses 6 lanes of 32 and runs at 13.2x its memory floor - the worst ratio anywhere in the library. Giving each
simdgroup 32/(dims+2) rows at once, with same-cluster collisions resolved by rank so it stays deterministic,
raises utilisation to 30/32 and measures **1.36x at 4 dims, 1.22x at 6, and 0.76-1.03x everywhere else**.
The lanes were never the constraint: the existing kernel hides memory latency by loading 8 rows ahead, and
the multi-row version spends that parallelism on width instead. Winning would need both, which is not worth
it for 0.6% of the suite.

**A per-slice maximum |x|^2 in the margin kernel.** It recomputes |x|^2 for every row on every pass, a
second full read of X. Substituting a single maximum is safe by construction - a larger value only widens
the error window, so it can never discard the true nearest - and labels were indeed identical. Worth
**1.12x on the margin kernel, 1.7% of the GIST step.** Not worth a cached norm and its staleness risk.
Note the window widens a lot for a typical row (GIST's max |x|^2 is 9.3x its mean) and the result still did
not change, which says something about how loose the bound already is.

**Cache-blocking by dispatching smaller chunks.** Covered above: monotonically worse, because each chunk
costs another 150 us round trip.

The pattern across all four: this library is no longer losing time to things that can be fixed by choosing
better constants or better instructions. What is left is either at a hardware ceiling (the 15.7 TFLOP/s
multiply) or below the noise floor of the measurements.

## Low-precision simdgroup matrices do not raise the ceiling (September 25, 2026)

Tested at Codex's suggestion (`meta/ai/CODEX.md`), as the kill switch for a low-precision fused assignment
kernel: the no-load ceiling of `simdgroup_multiply_accumulate` is 15.8 TFLOP/s in fp32, 16.0 in fp16, 15.9 for
fp16 or bf16 inputs into fp32 accumulators, and 14.1 in bf16 throughout. The instruction is a single ~16
TFLOP/s unit regardless of type. MLX reaches 36-52 TFLOP/s through hardware custom Metal kernels cannot
address, which settles the high-dimensional question for this library: fusion cannot recover the multiply
throughput it gives up, at any precision.

## The approximate path had a third of its time in avoidable passes (September 25, 2026)

At 960 dims the `exact=False` pass read X three times - multiply, a separate kernel for the distance to the
chosen centre, and accumulation - and the fp16 multiply had been rejected on a test that converted X every
call. Fixing both: GIST 500k x 960, k=1024, 41.1 -> 27.3 ms a pass, 2.68x on a full fit against exact (was
1.61x), recall@10 unchanged (92.43% vs 92.44%). fp16 is gated by a Cauchy-Schwarz bound on |x||c| against a
quarter of float16's range; SIFT fails it and stays fp32, as it must - every SIFT row's |x|^2 overflows.

The correction to the record: "fp16 is worth 6%" was a measurement of my test harness, not of fp16.

## Optimal at 960 dims: the accounting that ended the loop (September 25, 2026, AC power)

GIST1M, first 500k rows, k=1024. Floors from the ceilings in `MACHINE-PROFILE.md`; measured stage by stage
with a sync after each, best of 5-7.

**Approximate pass (`exact=False`): 26.7-27.1 ms, floor ~23.4 ms, gap ~14%.**

| stage | measured | floor | note |
|---|---:|---:|---|
| fp16 multiply + argmin, chunked, no per-chunk eval | 19.5 ms | 19.5 | 17.4 ms of multiply at MLX's 56.6 TFLOP/s + 2.1 ms to read the 1 GB fp16 product once; the product write is inside the matmul number |
| counting sort | 0.70 | ~0.5 | |
| segment kernel: distance + compensated sums, fp16 X | 2.36 | 1.96 | one fp16 read of X; **83% of bandwidth floor** |
| segment reduce | 0.22 | 0.2 | |
| totals to float64 on host | 0.86 | ~0.7 | best alternative found: 0.70 (mx float64 add on the CPU stream) |
| centre update (numpy, float64) | 1.61 | ~1.2 | float32 division would be 1.24, in approximate mode only |

The two host items are 0.16 + 0.37 ms = 2% of the pass, each below the ~3% multiply variance and far
below the ~25% bandwidth median drift measured the same hour. Neither is worth the code they would need.

Two things that could have moved the floor were checked and do not: cutting the assignment into
cache-resident chunks (8-16 MB of product) without per-chunk syncs gives 19.4-19.9 ms at every chunk size
from 4096 to 262144 rows - no gain and no loss, because MLX already pipelines unsynced chunks; and the matmul
rate at our K=960 tops out at ~57 TFLOP/s regardless of M or N (63 needs K=4096, which is not our problem).

**Exact pass: 90.3 ms.** The multiply runs at 14.9 TFLOP/s against a 15.7 ceiling for the only instruction
an exactness bound can be built on (three independent measurements: the no-load ceiling, the MAC-count
ablation, and Codex's fused kernel built and lost at every dimension). The margin kernel's per-row |x|^2
recompute is ~4 ms (4.4%), the only identified slack, and a per-slice maximum instead was worth 1.12x on
that kernel - 1.7% of the pass.

**Scope.** Optimal *given that* custom Metal kernels cannot reach the multiplier `mx.matmul` uses, and
`mx.matmul` cannot fuse a reduction into its output. If MLX exposes a fused matmul-reduce, or a future
Metal exposes the fast multiplier, the approximate floor drops by up to the 2.1 ms product read plus
whatever the argmin then costs on chip, and this section should be redone.

**Verdict.** At 960 dims both modes are at their respective units' limits, with residuals inside
measurement resolution. Further passes here would be measuring noise. The loop stops.

## Input limits, measured

Behaviour on degenerate input, worth knowing before trusting a result:

| Input | Behaviour |
|---|---|
| k > rows | raises ValueError |
| 1-D input | raises ValueError |
| k == rows, k = 1, single row | correct, inertia 0 where expected |
| all points identical, zero-variance column | correct |
| float64 or integer input | converted, correct |
| k up to 65,536 | works |
| **NaN in the data** | **silently returns nan, no error** |
| **inf in the data** | **silently returns nan, no error** |
| **coordinates above ~1e18** | **silently returns nan**: squared distances overflow float32 above ~1.8e19 |

The last three are the gap. Values up to 1e15 are fine (inertia 5.0e33), so the failure is confined to
genuinely extreme magnitudes, but it is silent in all three cases - a caller gets nan centres rather than an
error. Standardising the features, which the file workflow does by default and the README recommends for
other reasons, avoids all three.

Capacity per GPU slice, set by uint32 indexing in the kernels (row * dims must fit): 100M rows up to 42
dims, 33.5M at 128 dims, 4.47M at 960 dims. Above that the data is held as several slices, which the API
accepts as a list of MLX arrays; the billion-row regression uses this.

## Reuse audit: what broke when the library was used from outside (September 25, 2026)

A 104-agent adversarial review of the library as something other projects would import, covering the API,
the exact and approximate paths, accumulation, the claims in the docs, robustness and reuse. The kernels came
through clean: every exact path (rows, pairs, both tile variants, tiles1 register and reload, the cascade with
stale bounds and 55-62% label churn, ragged n, k and d) matched a float64 reference with zero wrong labels
and identical tie-breaking; the counting sort matched numpy's stable argsort at every k and n boundary tried;
determinism held bit-for-bit across processes on twelve default-path shapes. The defects were all in module
state and dispatch, and all in the fit-here-predict-there pattern that a single-dataset suite never exercises:

- **Blocker, fixed.** `_approx_state`, `_cascade_state` and `_tiles1_slow` were keyed on `id(parts[0])`.
  Python recycles an id the moment the array is freed, so `fit(A)` then `predict(B)` on a same-shaped
  array handed B the fp16 copy of A (97% wrong labels, 3x inertia) or A's cascade prefix and last labels
  (mislabels or a "please report this" error that blamed the user's data). Entries now carry weak references
  to every part and are checked by identity; a dead or different reference means a rebuild. `tests/reuse.py`
  reproduces the failure on the old code (1.5-3% label agreement) and passes on the new.
- **Major, fixed.** A feature-count mismatch between fit and predict aborted the interpreter: the fused
  kernel was sized from the centres, the data was wider, the threadgroup allocation overflowed, and the
  Metal error surfaced inside numpy's buffer request, where a C++ exception cannot propagate. Every
  MLX-to-numpy conversion now evaluates first (`_host`), which turns any Metal error into a RuntimeError the
  fallbacks can catch, and the estimator and `assign_mlx` both check the feature count.
- **Major, fixed.** A single MLX array over 2^32 elements (5M x 960 = 19 GB) wrapped the kernels' uint32
  indexing silently; the estimator now slices MLX input like numpy input, and `assign_mlx` refuses an
  oversized part. `method="tiles"` forced by name with k or d not a multiple of 8 was silently wrong; it
  now raises.
- **Open (the fp16 guard bounds the range of the product, not its precision).** With `exact=False`,
  uncentred data with |x||c| around 1e4 passes the Cauchy-Schwarz guard and gets 63% wrong labels; data
  below ~1e-5 collapses into fp16 subnormals (98% wrong); one finite coordinate above 65504 converts to
  inf and surfaces as a misleading error. All three need the guard to look at the operands' magnitude
  spread, not just the product's range. Also open: `_approx_prepare` peaks at 2x the dataset and keeps the
  fp16 copy after X is deleted; with `APPROX_FP16_ACCUMULATE=True` the inertia and centres come from the
  fp16 copy (7e-7 and 2.7e-4 relative on GIST) while a docstring says otherwise; the docstring's per-pass
  error figures are 8x too optimistic for the fp16 path (1.2% labels, 2.4e-5 inertia measured on GIST).

## The per-pass benchmark hides the largest lever left (September 25, 2026, AC power)

At 300k x 960, k=1024 the exact path spends 49.8 ms of a 57.6 ms iteration in the GEMM+argmin (11.8 of the
15.7 TFLOP/s ceiling); everything else - grouping, sums, centre update - is 7.8 ms. Any implementation that
runs the full n x k x d GEMM every iteration sits at this wall once the data is large enough for launch,
sync and materialised-matrix costs to have amortised away; the wins recorded above are all of that second
kind, and they do not move the wall.

The masked fact: after iteration 1 almost none of that GEMM is needed. On GIST 300k x 960, k=1024 from a
random-row start, with a 20k-row sample and the full previous-iteration distance matrix (prototypes/bounds_probe.py, run from the repo root):

| iter | ms | labels changed | max centre drift | median drift | n x k distances needed (per-centre bounds) |
|---|---|---|---|---|---|
| 0 | 144 | 52% | 6.21 | 0.77 | 93% |
| 1 | 58 | 21% | 1.21 | 0.15 | 12% |
| 2 | 58 | 13% | 0.84 | 0.08 | 2.9% |
| 3 | 57 | 9% | 0.59 | 0.05 | 1.3% |
| 5 | 57 | 6% | 0.54 | 0.03 | 0.55% |
| 8 | 56 | 4% | 0.20 | 0.02 | 0.30% |
| 13 | 57 | 2% | 0.66 | 0.01 | 0.19% |

"Needed" counts the centres j with d_old(x, j) - drift_j < d_old(x, own) + drift_own, before tightening the
upper bound with one exact distance: the pairs the triangle inequality cannot rule out. A single global
bound (Hamerly) rules out ~0% of points here, because every iteration a few centres move 20-40x the median;
it has to be per-centre (Elkan) or per-group (Yinyang) bounds. Elkan's bounds are n x k float32 - 1.2 GB at
this shape, impossible at 100M rows - so the scalable form is Yinyang's n x t group bounds with t around
k/10. It fits the exact path: stored float32 distances and drifts padded by the same gamma bound the tie
tolerance uses keep the labels provably exact. It does not make the approximate path exact; its stored
distances are fp16-derived.

The honest ceiling: bounds remove only the GEMM. Iterations from 2 on would drop from ~57 ms to the 7.8 ms
floor plus bound maintenance, one exact distance per point (a re-read of X, ~2.4 ms at 490 GB/s) and the
sparse residual - realistically 12-18 ms, so 3-5x per later iteration on the exact path and ~2x on the fp16
path. Iteration 0 and k-means++ initialisation are untouched and become the dominant cost. The suite and the
per-pass comparisons measure seconds per pass over 5 fixed iterations, where the bound is worth 0 at
iteration 0 and ~88% at iteration 1, which is why none of this showed. A fit to `tol` on GIST runs 20-50
iterations. Built the same day; the next section is the record.

## Per-centre bounds between iterations: built (September 25, 2026, AC power)

Section 5 of core.py. The design measurement first, because it overturned the plan. Group bounds (Yinyang: one
lower bound per row per group of centres, so n x t memory) were the intended form for memory reasons.
Measured on the same GIST 300k x 960, k=1024 sequence, with the row's own distance recomputed exactly each
iteration (prototypes/bounds_group_probe.py), the fraction of the n x k pairs that still have to be visited:

| iter | per-centre bounds | groups of 8, random | groups of 8, spatially coherent | groups of 16, coherent |
|---|---|---|---|---|
| 1 | 2.5% | 70% | 51% | 68% |
| 2 | 0.7% | 40% | 27% | 45% |
| 5 | 0.24% | 13% | 7.7% | 18% |
| 11 | 0.14% | 1.9% | 1.3% | 3.4% |

A group inherits its biggest mover, and drift is heavy-tailed (a few centres move 20-40x the median every
iteration), so grouping throws away most of the gain. The bounds are therefore per centre - Elkan's - with the
n x k float32 array that implies, and the path is offered only where that fits (BOUNDS_MAX_BYTES, 16 GB
including one chunk in flight: 3.9M rows at k=1024). The state is dropped the moment any of its arrays is freed,
so fit() on numpy input leaves nothing behind.

How it runs. A full pass records labels, own distances and centres (no n x k memory). The next full pass
first asks a strided sample of rows (8k), given fresh bounds against the recorded centres, what fraction of
the pairs a step would visit; only under BOUNDS_MAX_VISITED does that pass also build the bounds for every
row - one extra kernel over the dot chunks the tiles path computes anyway (kmeans_lb_init, one coalesced
read per chunk; as a chain of MLX elementwise ops it was seven passes and cost more than the multiply). Each
iteration with bounds in hand: a strided count pass says how many pairs they would visit; above the
threshold they are dropped and the full pass runs, which is what makes an n_init restart or a large early
drift safe. Below it, one kernel per row: subtract each centre's drift from its bound, recompute the own
distance exactly (so the inertia is exact and the upper bound tight), gather the centres whose bound fell
below it into a 32-bit mask with simd_sum, walk it with ctz, and visit only those. The accumulation kernels
run unchanged on the labels it produces. Only lloyd_step records or builds; a one-shot predict never
allocates n x k.

Two things the first version got wrong, both caught by the three-way comparison and the suite the same day:

- It built the bounds on every full pass, on speculation. On isotropic high-dimensional data (500k x 384,
  k=1024, Gaussian blobs) the margins between centres are a few percent of the distance - smaller than the
  drift - so the bounds never paid, every pass rebuilt 2 GB, and 10 iterations took 1888 ms against 407.
  Hence the predictor, and a back-off: a build that lasts a step or less makes the next question wait
  1, 3, 7 iterations. That case now takes 430 ms.
- Building the bounds from MLX's matmul (x @ C.T) instead of the tiles path's own dot kernel gave 1386 wrong
  labels in an 80k-row suite case, at 8e-4 excess distance: MLX's float32 product carries ~630x the error
  of the direct sum (the finding that kept the exact path off matmul in the first place), so a bound padded
  for the direct sum's error and built from it is not a bound. The build uses the simdgroup dot kernel,
  with the direct-distance kernel for the rows that do not fill a tile.

Costs, GIST 300k x 960, k=1024 (theta_sweep, rebuilds disabled so the visited fraction drifts on its own):
a bounds pass costs about 20 ms + 3.8 ms per percent of pairs visited (0.84%: 22 ms; 4.9%: 37 ms; 21%: 98 ms;
47%: 201 ms), against 57 ms for the plain pass and 75 ms for the pass that also builds the bounds. Left alone,
the visited fraction falls monotonically as the run converges (47% right after a random-row init, 5% by
iteration 9, 0.8% by iteration 39); after a rebuild it starts near 0.3% and creeps up as unvisited bounds decay,
which is what the threshold catches. The threshold itself hardly matters: a fit to tol=1e-4 (26 iterations)
took 1.05-1.15 s at every value from 3% to 20%; it stays at 5%.

Results, same machine, old tree against new, interleaved:

| | plain | bounds | |
|---|---|---|---|
| GIST 300k, k=1024, fit to tol=1e-4 (26 iterations both) | 1.85 s | 1.05 s | 1.75x |
| GIST 1M, k=1024, 5 iterations | 0.89 s | 0.74 s | 1.20x |
| GIST 1M, k=1024, 20 iterations | 3.51 s | 1.62 s | 2.17x |
| GIST 1M, k=1024, one iteration in steady state | 175 ms | 50-63 ms | 2.8-3.5x |

The 5-iteration row is the suite's metric, and it moves least: iterations 0 and 1 are full passes (the
first drift after a random-row start is huge), iteration 2 is the full pass that builds, and only 3 and 4
run on the bounds - which is exactly why this lever was invisible to the per-pass benchmark.

Exactness. tests/accuracy.py has three multi-step cases on the path (60k x 960 k=1024 with unrelated centres
injected at step 6, three slices at 256 dims, and one with nine empty-cluster relocations): at every step the
labels are float64-optimal to the suite's float32 tolerance and the centres match the float64 reference step
within 1e-7. What changes is the tie rule: a row keeps its label unless a visited centre is strictly nearer in
the computed distance, and the bounds kernel sums a distance lane-parallel where the tiles kernel sums it
sequentially, so a float32-level tie can land differently. On the 300k sequence the two paths agreed on every
label for four iterations, then one row flipped (squared-distance gap 7e-7 relative) and the runs followed
different, equally valid trajectories to final inertias 5e-5 apart. On the 1M sequence they agreed to seven
digits through 20 iterations. Both paths are deterministic run to run.

This reconciles with the earlier finding that "triangle-inequality pruning is useless at 960 dims" (78% of
centres survive): that was the centre-to-centre test |c - c_a| > 2 d_a inside one pass. The bound that works
is the one carried between passes - last iteration's distance to each centre, minus that centre's drift -
which is a different quantity, and at 960 dims it removes 97-99% of the work.

Not done, in the order it would pay: the 3.8 ms per percent is centre reads from cache, one full row per
visit, and would fall several-fold if the visited pairs of eight rows were evaluated as simdgroup tiles;
fp16 bounds (halves the 4 GB at 1M rows and the rewrite traffic; needs round-toward-zero conversion, and an
unvisited bound loses a half-ulp per iteration); building the bounds inside the margin kernel instead of a
second read of the dot chunk (+27 ms per rebuild at 1M rows); and the count pass's sample (1.5 ms at 300k)
could be smaller. Below 256 dims the path is not offered and has not been measured.

## Where a low-dimensional pass loses to MLX's own primitives (September 25, 2026, AC power)

2M x 50, k=64 (the single-cell shape), one pass, same data and start throughout:

| | ms | against a floor of |
|---|---|---|
| reading X once (400 MB at 490 GB/s) | 0.82 | - |
| our assignment, tiles1 (exact) | 2.11 | one read |
| our accumulation, sorted (deterministic) | 2.27 | one read |
| our accumulation, atomic (opt-in, not deterministic) | 3.24 | one read |
| **our pass, auto** | **4.1-4.7** | 0.82 fused, 1.64 unfused |
| MLX matmul + argmin, float32 | 9.33 | - |
| MLX scatter-add (`zeros.at[labels].add(x)`) sums + counts | 1.63 | one read |
| one-hot float16 matmul for sums + counts | 4.02 | one read |

Each of our two stages is 2.5-2.8x its floor, and the fused single-read kernel cannot apply: it keeps a
private Kahan accumulator per thread in threadgroup memory, which caps it at k*(dims+2) <= 128 (here 3328).
MLX's scatter-add does the accumulation in 1.63 ms - faster than our own atomic tile, which is the same idea
with a worse kernel - at the price of atomics (not bit-reproducible run to run), and a float16 matmul does
the assignment in about a millisecond at the price of exactness. A pass built from those two primitives
takes about 2.7 ms here, 1.5x faster than ours; on this shape the exact, deterministic pass is paying
about 1.4 ms for each of its two guarantees. Closing it means one exact kernel that assigns and accumulates
in a single read of X at this k*w, with accumulators shared across a simdgroup rather than private per
thread - not a tuning of either existing stage.

What the two guarantees cost, measured on four shapes (same data and start; assignment exact = the auto
kernel, fp16 = our approximate path forced; accumulation deterministic = auto, scatter-add = MLX's
`zeros.at[labels].add(x)` for sums, counts and inertia):

| shape | pass | assign exact | assign fp16 | accumulate det. | accumulate scatter-add | both dropped |
|---|---|---|---|---|---|---|
| single-cell 2M x 50, k=64 | 4.08 | 1.92 | 1.47 | 2.32 | 1.78 | ~3.25 |
| logs 10M x 32, k=256 | 22.7 | 17.5 | 23.3 | 5.19 | 6.49 | slower |
| geo 10M x 6, k=8 | 2.08 | 0.82 | 3.05 | 3.71 (fused pass reads once) | 12.2 | much slower |
| SIFT1M 1M x 128, k=1024 | 21.7 | 22.6 | invalid (fp16 guard fails: 0.06% agree) | 1.93 | 1.49 | 21.3 |

So determinism costs at most 0.5 ms of a pass (single-cell, 13%; SIFT, 2%) and is free or better at logs
and geo, where the deterministic kernels beat atomics outright; exactness costs 0.45 ms at single-cell, is
free below and above it, and cannot be dropped on SIFT at all. A "result-only" pass with both dropped is
3.25 ms at single-cell - still behind the 2.7 ms that MLX's own primitives reach there - and slower than
the exact, deterministic pass on every other shape. The single-cell gap is structural (two reads of X and
the host round trips between them), not the price of the guarantees.

## The fused single-read kernel, second attempt (September 25, 2026, AC power)

Section 1b of core.py, `method="onehot"`. Built for the low-dimensional gap above: one kernel that assigns a
row and adds it to its cluster's sums without a second read of X and without the host round trips between
the two stages. The design that worked holds the per-cluster sums as simdgroup 8x8 tiles in registers and
adds each 8-row batch with one-hot multiplies on the matrix unit - sums (k x W) += onehot^T (k x 8) @ rows
(8 x W) - with the rows' tiles loaded straight from X (just read for the assignment: cache) and a 64-float
tail tile carrying the ragged dims plus [1, best], so counts and inertia come out of the same multiply.
Threadgroup memory is 2.3 KB per simdgroup at k=64. Partials per simdgroup, Kahan reduce in index order:
deterministic, and the centres came out bit-identical to the two-stage path on every shape tried. Labels
and best are the rows kernel's, bit for bit.

Three designs were measured before this one, all exact:

| design | 2M x 50 k=64 | why |
|---|---|---|
| lane-owned clusters, rows broadcast with simd_shuffle | 32 ms | O(32 x dims) per row whatever k is, and the two accumulators plus the row spill the register file |
| one-hot tiles, 8 rows staged by 8 lanes, one-hot zeroed per batch | 3.8 ms (accumulate alone 2-3) | the staging and zeroing, not the multiplies: sorted labels with 7 of 8 blocks skipped still cost 1.3 ms |
| one-hot tiles, 32 rows staged at once, 15 KB per simdgroup | 5.8 ms | occupancy: see the threadgroup-memory table in MACHINE-PROFILE (measured for this) |
| **one-hot tiles, X tiles from cache, 2.3 KB per simdgroup** | **4.2 ms, parity** | assignment-bound (below) |

The threadgroup-memory table is the durable result: streaming bandwidth holds to about 64 bytes of
threadgroup memory per resident thread and halves by 128 - which is the measured reason a per-simdgroup
accumulator cannot live there, and the reason the first fused kernel is limited to k x (dims+2) <= 128.

Measured on the benchmark's own generator, a full lloyd_step, best of three medians after a discarded first
one (the first median after a dataset is created runs 1.5-2x slow - page-in or clock - and an earlier sweep
without the discard read every 2M-row shape as a loss):

| shape | two-stage | onehot | | shape | two-stage | onehot | |
|---|---|---|---|---|---|---|---|
| 20M x 8 k=16 | 8.29 | 2.79 | 2.97x | 2M x 12 k=128 | 2.72 | 2.33 | 1.17x |
| 10M x 12 k=32 (satellite) | 4.88 | 2.83 | 1.72x | 2M x 12 k=64 | 1.49 | 1.39 | 1.07x |
| 2M x 50 k=32 | 3.97 | 2.47 | 1.61x | 2M x 24 k=96 | 3.39 | 3.39 | 1.00x |
| 0.5M x 50 k=64 | 1.98 | 1.40 | 1.41x | 2M x 50 k=64 (single-cell) | 4.15 | 4.21 | 0.99x |
| 2M x 90 k=16 | 5.14 | 3.74 | 1.37x | 2M x 20 k=48 | 1.68 | 1.76 | 0.96x |
| 2M x 30 k=64 | 3.36 | 2.52 | 1.34x | 2M x 24 k=32 | 1.50 | 1.62 | 0.93x |
| 2M x 62 k=24 | 3.26 | 2.51 | 1.30x | 10M x 4 k=256 (geo-trips) | 8.53 | 7.57 | 1.13x |
| 10M x 6 k=8 | 1.44 | 1.15 | 1.25x | 2M x 12 k=256 | 3.68 | 5.46 | 0.67x |
| 2M x 40 k=16 | 1.90 | 1.58 | 1.20x | 2M x 8 k=256 | 2.89 | 4.85 | 0.60x |

So the path is offered up to k=128 (never below 0.93x measured; k=256 loses at 2M rows and its 1.13x on
geo-trips is left on the table), dims up to 96, k x dim-tiles up to 640 (the accumulator tiles are
registers). In the suite: satellite 5.1 -> 3.2 ms per pass; single-cell unchanged at parity.

Why single-cell is parity and not the 1.5x hoped for: the assignment. Sustained (40 launches queued), the
fused kernel costs 1.9x the bare rows assignment at 2M x 24 k=32 (0.93 vs 0.50 ms) - the one-hot multiplies
and barriers cost about what the rows loop does - and at 2M x 50 k=64 tiles1 assigns in 1.8 ms where the
rows loop needs 2.4; the one-hot accumulation saves exactly that over the sorted path. What would move it is
tiles1's assignment (simdgroup dot products with the candidate check) inside this loop, where its X tiles
are already loaded; that is a new kernel, and the same one that would close geo-trips (5.6x off its floor,
all of it in the scalar assignment at k=256, d=4).

Floors are now a column in BENCHMARKS.md (`bench.py --report`): one read of X at 490 GB/s or 2*rows*k*dims
at 15.7 TFLOP/s, whichever is larger, and the measured pass over it. Today: geo-trips 7.1x, single-cell
5.4x, satellite 3.2x, logs 2.7x, SIFT 1.4x, GIST 1.4x. The GEMM-bound pair are at the hardware within the
accumulate and the candidate check; the low-dimensional four are the named gap, and the assignment is most
of it.

## What the machine allows, per config (September 25, 2026)

The floor column is the simplest exact pass on the resources our kernels reach. The other direction - what the
hardware measurably does, whichever path reaches it - is `benchmarks/possible.py`, from the rates in
MACHINE-PROFILE: DRAM 490 GB/s, simdgroup matrix unit 15.7 TFLOP/s, scalar FMA 13.8, `mx.matmul` on each
config's own GEMM, host round trip 150-200 us, launch 5 us. Milliseconds for one Lloyd iteration, and how far
the measured pass is from each bound:

| config | measured | exact, our kernels | exact via the fast matmul | converged regime, per-centre bounds fp32 / fp16 | result-only (fp16 matmul) | round-trip floor |
|---|---|---|---|---|---|---|
| geo-trips 10M x 4, k=256 | 9.2 | 1.30 (7.1x) | same | same (bounds cost more than the GEMM) | same | 0.18 |
| satellite 10M x 12, k=32 | 3.2 | 0.98 (3.3x) | same | same | same | 0.18 |
| logs 10M x 32, k=256 | 27.8 | 10.4 (2.7x) | same | same | same | 0.18 |
| single-cell 2M x 50, k=64 | 4.4 | 0.82 (5.4x) | same | same | same | 0.18 |
| SIFT1M 1M x 128, k=1024 | 22.8 | 16.7 (1.4x) | 13.4 (1.7x) | 16.7 / 8.4 (1.4x / 2.7x) | 8.1 (2.8x) | 0.18 |
| GIST1M 1M x 960, k=1024 | 176; 50-63 converged | 125 (1.4x) | 47.8 (3.7x) | 16.8 / 8.4 (3.0x / 6.0x on the converged 50) | 34 (5.2x) | 0.18 |

How to read it:

- **Below 128 dims the machine has exactly one fast resource for this problem: DRAM at 490 GB/s.** The matrix
  hardware through `mx.matmul` is slower than a simdgroup kernel there (1-14 TFLOP/s), per-centre bounds
  move more bytes than the distances they save (n x k x 8 per iteration: 20 GB at logs), and float16 buys
  nothing because the pass is not compute-bound. The bound is one read of X, and the four low-dimensional
  configs are 2.7-7.1x from it. All of that distance is ours: two reads instead of one, the scalar
  assignment at 18-40% of the FMA peak, and the host round trips between stages.
- **Above 128 dims the machine has three resources we do not use.** The fast matmul (41 TFLOP/s float32 at
  GIST) would put an exact pass at 48 ms instead of 125 if its candidates could be verified cheaply - the
  "matmul hybrid abandoned on precision grounds" is worth reopening with a candidate check sized to its
  actual error rather than the direct sum's. Per-centre bounds in the converged regime are bandwidth-bound
  on their own traffic (17 ms float32, 8.4 float16 at 1M x 1024), against 50-63 ms measured: 3-6x is left
  there, most of it the visited pairs evaluated a row at a time. And float16 bounds halve that traffic.
- **Below about a million rows the round trip is the floor**, not the data: 0.18 ms per iteration if the
  convergence check runs on the host, which is 4x one read of 100k x 50. A GPU-resident loop (no sync per
  iteration) is the technique; nothing in the suite is small enough to show it.
- What no per-iteration bound sees: the iteration count. Initialisation and time-to-tolerance are the other
  half of a fit, and the suite does not measure them.

