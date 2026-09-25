# Codex recommendation: one bounded high-dimensional flash experiment

Date: 2026-09-25

## Scope and clean-room boundary

This recommendation is based only on this repository's implementation, notes, commits, and measurements. I did
not open, clone, search for, or inspect the source of Flash KMeans. Do not treat any hypothesis here as a claim
about how that project works.

Satya intentionally authorized relaxing assignment accuracy to match high-dimensional speed. The current
`exact=False` path is therefore a legitimate product mode, not an accidental regression: it uses MLX matmul,
chooses labels from the approximate expanded distance, and recomputes the selected distances so reported inertia
describes the labels actually returned.

## Current conclusion

The repository already uses the FlashAttention principle where it wins: fused assignment avoids the global
`rows x clusters` distance matrix at moderate dimensions; candidate distances are recomputed rather than stored;
the cascade spends more arithmetic to prune global traffic; and accumulation keeps segment state in registers.

The high-dimensional approximate path is not fully flash-fused. It still performs:

```text
MLX matmul -> materialized dot-product chunk -> one custom norm-adjustment/argmin read
```

That is much better than walking the product four times, but it still writes and rereads the product. Previous
fp32 fused variants were correct and slower because they combined two disadvantages: the custom fp32
`simdgroup_multiply_accumulate` ceiling is about 15.7 TFLOP/s versus roughly 29-36 TFLOP/s for MLX matmul at the
relevant shapes, and the scan left most SIMD lanes idle. Fusion alone cannot overcome losing about half the
multiply throughput at 960 dimensions.

## The one fresh attempt I recommend

Build a stand-alone **low-precision, fully occupied, online-argmin assignment microkernel**, initially for
L2-normalized embeddings and spherical k-means. Do not integrate it into `KMeans` until it wins the microbenchmark.

For normalized rows and normalized centers, assignment is simply:

```text
nearest center by cosine similarity = argmax(x dot c)
```

This avoids squared norms, expanded-distance cancellation, and fp16 norm overflow. It is the cleanest place to
test whether low-precision custom matrix instructions can combine MLX-class arithmetic throughput with flash-style
elimination of the global product.

### Kernel shape

For each row tile, stream center tiles and carry only `(best dot, center id)`:

```text
for each row tile:
    running_best = -infinity
    for each center tile:
        scores = low_precision_matrix_multiply(row_tile, center_tile)
        tile_best = all-lane segmented_argmax(scores)
        running_best = max(running_best, tile_best)
    write one label and score per row
```

Start with a 16-row x 16-center result tile. Assign two SIMD lanes to each row: each lane reduces eight centers,
then one `simd_shuffle_xor` combines the pair. All 32 lanes participate, unlike the rejected design in which about
eight lanes scanned while the other 24 waited. Also test an 8 x 8 layout where each lane owns two entries and the
row reduction uses segmented shuffles.

Sweep 16, 32, and 64 centers per reduction. Computing several independent center tiles before reducing may
amortize barriers, but stop increasing the batch as soon as matrix accumulators spill. Prior measurements show
that indexed multidimensional accumulator arrays and runtime-indexed candidate arrays spill catastrophically;
use named/flat accumulators in every prototype.

### Arithmetic variants to measure first

1. fp16 inputs with float32 accumulation, if the Metal matrix API supports that combination efficiently.
2. bfloat16 inputs with float32 accumulation, if available; its exponent range avoids fp16's sharp overflow.
3. fp16 accumulation only as an explicitly approximate baseline.

Before writing the full kernel, measure the no-load custom matrix-instruction ceiling for each type. MLX's
52 TFLOP/s fp16/bfloat16 result does not prove that a public custom Metal kernel can reach it. If the custom
low-precision instruction remains near the measured fp32 ceiling, terminate the experiment immediately.

For non-spherical Euclidean data, a later prototype may apply one global power-of-two scale to rows and centers
before low-precision assignment. In exact arithmetic this preserves distance ordering. Low-precision rounding can
still change near ties, which is acceptable only under `exact=False`. Accumulate and update centers from the
original float32 data.

## Microbenchmark and kill criteria

Use fixed centers and synchronized timing on approximately:

```text
200K-1M rows x 960 dimensions, k=1024
```

Compare against the current `mx.matmul + _APPROX_ARGMIN_SRC` path. Record assignment time, effective multiply
throughput, peak memory, differing labels at identical centers, relative inertia, and downstream IVF recall for
GIST1M. Do not compare two independently converged fits as a label-accuracy test.

Continue toward integration only if the prototype:

- exceeds roughly 25 TFLOP/s effective multiplication throughput;
- beats the current approximate assignment by at least 20%;
- shows no catastrophic register/scratch spill;
- has an explicit and acceptable label/inertia/IVF-recall tradeoff; and
- still improves complete fit time, not merely the assignment substep.

Stop if any of these fail. The existing fp16 experiment improved complete fit only 6%, showing that assignment can
cease to be the dominant cost after the current approximate optimization.

## Two measurements to take regardless

1. Produce a complete-fit decomposition at 128, 384, and 960 dimensions: conversion, initialization, assignment,
   accumulation, center update/host synchronization, final labels, and exact scoring. This determines whether any
   further assignment work has leverage.
2. Measure greedy k-means++ initialization as `k` grows. If it dominates at large `k`, k-means||, sampled
   initialization, or warm-started centers are more consequential than another assignment kernel. Those change
   the quality contract and must be reported as such.

## Recommendation

Authorize one narrow spherical low-precision flash microkernel experiment, with the ceiling measurement first and
the stop criteria above. Do not reopen broad fp32 fusion work: the repository already measured why it loses. If the
low-precision custom instruction cannot preserve the fast multiplier throughput, regard the current MLX matmul plus
fused argmin as the high-dimensional endpoint and move to the next MLX-focused project.

---

## Outcome (added by Claude, 2026-09-25)

**Stopped at the first kill criterion.** The no-load custom matrix-instruction ceiling, measured with the same
method as the fp32 one (no memory traffic, 8 and 16 independent accumulators, 16,384 simdgroups):

| inputs | accumulate | TFLOP/s | vs fp32 |
|---|---|---:|---:|
| float | float | 15.8 | 1.00x |
| half | half | 16.0 | 1.01x |
| half | float | 15.9 | 1.00x |
| bfloat | float | 15.9 | 1.00x |
| bfloat | bfloat | 14.1 | 0.89x |

`simdgroup_matrix` is one ~16 TFLOP/s unit whatever the type. MLX's 36-52 TFLOP/s comes from hardware that
instruction does not reach, so a custom low-precision kernel cannot combine MLX-class throughput with fused
argmin. No kernel was written.

The two measurements recommended regardless (500k rows, k=1024, 20 Lloyd iterations):

| dims | mode | convert | init | Lloyd | final labels | total | init share |
|---:|---|---:|---:|---:|---:|---:|---:|
| 128 | exact | 0.02 s | 0.09 s | 0.24 s | 0.01 s | 0.35 s | 25% |
| 384 | exact | 0.05 s | 0.12 s | 0.83 s | 0.04 s | 1.04 s | 11% |
| 384 | approx | 0.02 s | 0.10 s | 0.43 s | 0.02 s | 0.57 s | 17% |
| 960 | exact | 0.11 s | 0.26 s | 1.83 s | 0.09 s | 2.29 s | 11% |
| 960 | approx | 0.03 s | 0.25 s | 0.83 s | 0.04 s | 1.15 s | 22% |

Per pass: 960 exact is 82.7 ms assignment / 5.2 ms accumulation; 960 approx is 29.1 / 9.8; 384 approx is
16.3 / 4.5. (The 128-dim per-pass split is omitted: that path returns lazy arrays, so the timer measured graph
construction, not work.)

Greedy k-means++ against k, SIFT 500k x 128: 0.03 s at k=256, 0.06 s at 1024, 0.64 s at 4096 - four to six
Lloyd passes' worth throughout. It does not come to dominate.

So the closing branch of this recommendation applies: the current MLX matmul plus fused argmin is the
high-dimensional endpoint for this library. Work continues in `~/git/mlx-vsearch`.

### The kernel was then built anyway, to check the kill switch empirically

The ceiling alone left one argument open: fusion also avoids writing the n x k product, and at lower
dimensions that saving might outweigh the slower multiply (a rough model put the crossover near 450 dims). So
the recommended kernel was built as specified - `prototypes/fused_online_argmin.py` - and measured on GIST1M's
first 500k rows truncated to d columns, k=1024, labels only, per pass:

| d | exact (tiles) | current approx (mx.matmul + argmin) | fused, best variant | vs current |
|---:|---:|---:|---:|---:|
| 256 | 27.6 ms | 12.2 ms | 17.7 ms | 0.69x |
| 384 | 37.2 ms | 16.0 ms | 25.3 ms | 0.64x |
| 512 | 48.4 ms | 17.3 ms | 33.6 ms | 0.52x |
| 768 | 66.5 ms | 22.7 ms | 50.1 ms | 0.45x |
| 960 | 81.6 ms | 28.6 ms | 62.6 ms | 0.46x |

The best variant everywhere was fp16 inputs with 64 centres per reduction step. It is not badly built: at 960
dims it runs at 15.7 TFLOP/s, the instruction ceiling. MLX's path reaches 21-34 TFLOP/s *including* its argmin
pass, so the product traffic saved by fusion is worth less than the multiply throughput lost, at every
dimension measured. Fails both kill criteria (>= 25 TFLOP/s; >= 20% faster than current).

Side finding: at identical centres the fused kernel's labels differ from exact on 435-679 of 500,000 rows,
against ~2,000 for MLX's fp32 matmul path at the same scale - fp16 inputs with fp32 accumulation in a custom
kernel are about four times *more* accurate than MLX's matmul, consistent with the ~5,900 x eps error measured
for it earlier. Slower, but more faithful.

### Correction, later the same day: the approximate path was not at its endpoint

Satya pushed back on "the current MLX matmul plus fused argmin is the endpoint" - the external comparison
showed a whole iteration on GIST done in 26.1 ms against our approximate path's 41.4, and that path has no
exactness constraint to hide behind. Decomposing it stage by stage found two things:

- the pass read X three times: once for the multiply, once for a separate "distance to chosen centre" kernel,
  once for accumulation. The distance is now computed inside the sorted accumulator, where the row is being
  read anyway and the centre is fixed per segment. One pass gone.
- **fp16 had been wrongly rejected.** The earlier 1.06x measurement converted all of X to fp16 inside the call,
  every iteration - 5.5 ms of a 41 ms pass, eating most of the 1.44x the multiply gained. Converted once per
  dataset and kept, fp16 delivers: 49 against 34 TFLOP/s on GIST.

GIST 500k x 960, k=1024, per pass: 41.1 -> 27.3 ms (1.51x). Full fit 2.68x faster than exact (was 1.61x),
recall@10 92.43% against 92.44% exact. The guard is Cauchy-Schwarz on the norms; SIFT correctly refuses fp16.

So the endpoint claim was wrong, and the reason is worth keeping: the rejection rested on one measurement
whose setup charged a one-time cost to every iteration. The fused-kernel conclusion above still stands - the
custom-instruction ceiling is real - but "we cannot write a faster multiply" is not the same as "the path
around the multiply is done".
