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
