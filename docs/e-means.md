# e-means: an approximate, opt-in option beside exact Lloyd

`KMeans(algorithm="emeans")` and `mlx_kmeans.emeans_fit` run e-means, the mini-batch k-means with moving
averages described in *e-means* (arXiv:2611.09501, Meta FAIR; project page <https://e-means.github.io/>). It is
**approximate and opt-in**. The default, `KMeans(algorithm="lloyd")`, is the exact Lloyd iteration documented in the
README, and it stays the reference. This page covers the method in our own words, the choices we made where the paper
is silent, how the method runs on the GPU, the API, and what has and has not been checked or measured.

This implementation was written from the paper's text alone. We read no e-means source code, including the library
linked from the paper.

## The method

Exact Lloyd reads every row on every iteration and moves every centre to the mean of its rows. e-means trades that
for many cheap updates from samples.

- **State.** Each cluster *j* keeps a count `N_j` and a vector sum `S_j`. Its centre is always `S_j / N_j`.
- **Epochs and batches.** An epoch shuffles the *n* rows and cuts them into batches of *B*. The run takes a fixed
  number of epochs *T*, which gives `M = T * floor(n / B)` batch updates. There is no convergence test.
- **One batch.** Every row of the batch goes to its nearest current centre. That gives each cluster a batch count
  `n_j` and a batch sum `s_j`. Then every cluster, including those that received no rows, moves part of the way
  toward the batch:
  `N_j <- (1 - a) N_j + a n_j` and `S_j <- (1 - a) S_j + a s_j`.
- **Step.** The step *a* follows a cosine curve over all *M* batches. It starts at `a0` and falls to `0.1 a0`.
- **Revival.** After every fifth batch, any cluster whose `N_j` is below a tenth of the mean `N` is dead. Each dead
  cluster is revived by splitting a donor, drawn with probability proportional to `N`. The two children are the
  donor's centre scaled elementwise by `[1+δ, 1−δ, 1+δ, …]` and `[1−δ, 1+δ, 1−δ, …]`. Each child gets half the
  donor's count, and its sum is rewritten as centre × count.
- **Scaling laws.** *B* and `a0` are not tuned by hand; the paper fits laws for them:
  - `B = 0.035 k^0.13 n^0.88 T^0.81`, divided by 8 when *n* is above 2M, rounded to a power of two, and capped at *n*.
  - `a0 = min(0.824 k^-0.46 B^0.90 n^-0.45 T^-0.11, 1)`, using that *B*.
- **Initialisation.** The starting centres are *k* distinct random rows. Each starts with `N_j = 1e-10` and
  `S_j = N_j c_j`.
- **Output.** Each centre is the exact mean of the rows assigned to it during the last epoch, as they were
  assigned there.

For example, on SIFT1M at *k* = 1,024 and *T* = 3, the laws give *B* = 32,768 and `a0` = 0.696, which is 30 batches
per epoch and 90 updates in all.

## Our choices where the paper is silent

The paper leaves several details open. These are the choices this implementation makes, numbered as in the task plan.

1. **Batch counter.** *t* counts batches from 0 to M−1 across all epochs, and
   `a(t) = a0 (0.1 + 0.9 · 0.5 (1 + cos(π t / M)))`.
2. **Permutation and partial batch.** Each epoch draws a fresh seeded permutation. The last partial batch of each
   epoch is dropped, which matches `M = T floor(n/B)`. The `n − floor(n/B)·B` dropped rows differ from epoch to epoch.
   Because *B* is rounded to a power of two, the remainder can be large:
   - On the planned comparison shapes (1M rows, k = 1,024 to 65,536, T = 1 to 5) it is 0.06% to 1.7% of each
     epoch, and 8.3% at k = 1,024, T = 10 (B = 131,072, 7 batches).
   - At 1M or 2M rows and T = 25, most k values drop 21% (for example 1M rows with B = 262,144 gives 3 batches and
     a remainder of 213,568).
   - At 1M rows, k = 65,536 and T = 25, the law gives B = 524,288: one batch per epoch, 47.6% of the rows unused
     each epoch, so the last-epoch mean covers about half the data.

   Of 192 (n, k, T) shapes tabulated (n from 1e5 to 1e8, k from 256 to 65,536 with n/k ≥ 8, T from 1 to 25),
   9 drop more than 20%. The record's `dropped_per_epoch` and the verbose line give the count and fraction for a
   run. Whether to keep the partial batch instead is an open decision, not made here.
3. **Rounding *B*.** The /8 is applied before rounding. Rounding is in log2, `B = 2^floor(log2 B_raw + 0.5)` with
   halves going up. Then `B = min(B, n)`. `a0` uses the rounded *B*. A `batch_size` you pass replaces the law, and
   `a0` then comes from the law at your *B*.
4. **A cluster with N = 0** (possible when `a0` = 1) keeps its previous centre, so the code never computes 0/0.
5. **δ** is 2^−7, the bfloat16 machine epsilon that the paper names as its convention. The split scales
   coordinates, so how far the children move depends on where the data's origin is. On the GPU test's uncentred case
   (+1000 offset, data spread about ±4 per coordinate), each child coordinate moves about ±7.8, and that case ended
   with 117 of 256 clusters empty in the last epoch after 556 revivals; the float64 reference made the same
   decisions. That the displacement causes those empty clusters is reasoned, not measured, and the effect on SIFT
   and GIST was not measured. Whether to centre the data or change the split is open.
6. **Donors.** Donors come from live clusters only. They are drawn without replacement with probability
   proportional to `N`, as Gumbel-top-m: take keys `log N_j` plus Gumbel noise and keep the *m* largest. This has the
   same distribution as *m* successive proportional draws. If there are more dead clusters than live ones, as many
   are revived as there are donors, lowest index first, and the i-th dead cluster pairs with the i-th donor drawn.
   Revival continues in the last epoch.
7. **Last-epoch mean.** It is the sum of the last epoch's raw batch counts and sums, accumulated with compensation and
   divided once in float64. It covers the `floor(n/B)·B` rows the last epoch used. A cluster that received no rows in
   the last epoch keeps its final moving-average centre. This follows the paper's Alg. 1 line 12 literally, so a slot
   revived during the last epoch keeps the rows it was given at its old position and adds those given at its new one.
8. **No slow-EMA finalizer.** The optional slow-EMA finalizer (paper App. I) is not implemented.
9. **Whole epochs only.** *T* must be an integer. `epochs=1` is accepted. With one epoch, the "last epoch" starts
   from the random rows and contains every revival, and on the float64 reference that cost more than it gained.
   On 30k-row blobs (two seeds each), the last-epoch mean's full-data MSE was above that of the final moving-average
   centres:

   | Case | Last-epoch mean MSE above the final moving-average centres |
   |---|---|
   | overlapping, k = 128 | +0.14%, +0.21% |
   | separated, k = 64 | +0.61%, +2.3% |
   | B = 512, k = 256 | +0.27%, +0.29% |

   The gap was about +0.02% to +0.04% at T = 2, and at T = 3 the last-epoch mean was the better of the two. The
   paper's experiments use T ≥ 2. Whether to refuse `epochs=1`, or to return the moving-average centres for it, is
   an open decision; the code does neither.

The implementation adds these:

- **Revival timing.** The revival check runs after batch *t* when `(t + 1) % 5 == 0`, that is, after the 5th, 10th
  and later batches. The first batch uses the starting centres exactly.
- **Random numbers.** Each stream is keyed on `(seed, purpose, counter)` rather than read from one generator in
  sequence, so the GPU fit and the float64 NumPy reference draw the same values:

  | Draw | Code |
  |---|---|
  | Initial rows | `default_rng([seed, 1]).choice(n, k, replace=False)` |
  | Permutation for epoch *e* | `default_rng([seed, 2, e]).permutation(n)` |
  | Donors at batch *t* | `default_rng([seed, 3, t]).random(k)` |

  The donor stream draws one uniform per cluster whether or not any cluster is dead. A donor draw therefore never
  depends on how many earlier rounds revived. The permutation is computed as a shuffle of a uint32 `arange(n)`
  (int64 above 2^32 rows), which gives the same values as `permutation(n)` because the shuffle's draws do not depend
  on the dtype; the `--numpy` tests check the equality, so a NumPy change that broke it would show.
- **Sorted batches.** Each batch's row indices are sorted ascending. This does not change which rows a batch holds.
  It does make the gather from a list of slices a few contiguous runs, and it makes results bit-identical however
  the data is sliced.
- **Initialisation in the estimator.** `KMeans(algorithm="emeans")` starts from random rows, the paper's choice.
  `emeans_fit(init="k-means++")` uses the library's subsampled greedy k-means++ instead. You can also pass a
  (k, d) array of starting centres.
- **The approximate assignment** (`exact=False` / `assign="approx"`) uses the library's existing matmul and argmin
  kernels, with three differences from Lloyd's `exact=False`:
  - *Norms.* With the float16 product, each centre's norm is computed from the same float16-converted centre as the
    product, as the paper's App. J.1 requires (Eqs. 9-11). Lloyd's path takes it from the float32 centre.
  - *Sums.* The batch sums are accumulated from the float32 rows (App. J.2). Lloyd's path sums from the float16 copy.
  - *The float16 guard* is decided once per fit from a bound, not from the current centres. Between revivals a
    centre stays within the convex hull of its previous value and batch means, so its norm never exceeds
    R = max(max |x|, max |c0|). A split scales one centre by at most 1 + δ, and a centre is split at most once per
    check. So after all `M // 5` checks, |c| ≤ R (1 + δ)^(M // 5). The fit takes float16 only when both of these
    hold:
    - the repo's guard `|x| |c| < 65504 / 4`, with its 4× margin, for one split;
    - for the whole chain of splits, every converted coordinate and every product inside float16's range (65504).

    The second condition binds only past about 178 checks, where (1 + δ)^checks passes 4.

  One difference from the paper remains: the float16 × float16 product is returned in float16, so each dot product
  carries a rounding of about 2^-11 relative. Below that path's dimension thresholds the flag does nothing, as it does
  for Lloyd. The record's `approx_float16` says whether the float16 product ran.
- **Options not offered.** `spherical=True` and `accumulate="atomic"` raise a ValueError with
  `algorithm="emeans"`. `max_iter` and `tol` do not apply.

## How it runs on the GPU

Every batch runs on the GPU in float32:

1. **Gather.** The batch's rows are gathered from the data by a gather per slice. There is no host copy of the data.
2. **Assignment.** Each row goes to its nearest centre using the library's exact kernels. The method is chosen once
   per fit by shape:
   - `onehot`, a fused assign-and-accumulate kernel, for k ≤ 128 and dims ≤ 96;
   - `tiles1` from 32 dims (to 224 dims when dims is a multiple of 8) and k ≥ 64, after a single fallback probe on
     the first batch;
   - `tiles` for dims ≥ 80 and k ≥ 64, both multiples of 8;
   - `pairs` for dims ≥ 80 otherwise;
   - `rows` for everything else.

   The cascade and the per-centre bounds are not offered, because they keep state about the whole dataset across
   passes and a batch holds different rows each time.
3. **Batch counts and sums.** These come from a deterministic, compensated accumulator that stays on the GPU:
   - `lanes` where its tile fits on chip;
   - otherwise, the sorted-segment kernels with the segment layout planned in MLX ops (an argsort of the labels,
     integer scatter-add, cumsum and cummax). The plan is exact and deterministic, so there is no host read of
     counts. It is sized for the most segments there can be, `ceil(B / 256) + min(k, B)`.
   - `blocks` is available by name.
4. **Update.** `N`, `S` and the centres are updated in float32 on the GPU. Each batch is queued with
   `mx.async_eval`.
5. **Revival check.** Every fifth batch reads `N` (*k* floats) to the host for the donor draw. That read also keeps
   the lazy graph bounded.

**Where the host waits for the GPU.** Apart from the revival read, there are these waits:

- *At batch 0, once per fit.* `onehot` evaluates its first batch so that a GPU which refuses the kernel can fall back
  to rows. `tiles1`, when auto chose it, reads its fallback probe.
- *Per distance chunk, every batch.* `pairs`, `tiles` and the approximate path evaluate each chunk of the batch's
  distance matrix when there is more than one chunk, that is when `B · k · 4 bytes > 512 MB`. Each chunk is one
  wait, so for example GIST at k = 16,384 with B = 65,536 has 8 waits per batch.
- *Every batch, with `trace=`*, which is a test hook.

The rest of a batch, including the accumulation, the update and the revival writes, is queued without a wait.

Three other details:

- **Permutation.** Each epoch's permutation is built on the host in a worker thread: epoch 0's starts before the
  initialisation, and each later one while the previous epoch runs on the GPU. It holds 4 bytes per row, sorted in
  place within each batch. The record splits the time the GPU loop waited for it into `permutation_first_wait_s`
  (the part of epoch 0's permutation the initialisation did not cover) and `permutation_wait_s` (epochs 1 on).
- **Kernel wrappers.** The tiles wrappers build the transposed centre matrix on the GPU. Lloyd's wrappers build it
  from a host copy of the centres, which e-means would otherwise pay every batch.
- **Caches.** e-means touches none of the library's per-dataset caches, with one exception: the approximate path's
  float16 copy and norm bound, which are built once from the full data and shared with Lloyd's `exact=False`. No cache
  entry is ever keyed on a batch array.

## API

```python
KMeans(n_clusters, ..., algorithm="emeans", epochs=3, batch_size=None, exact=True)
  .fit(X)                 # centres from e-means; labels_ and inertia_ from Lloyd's final pass over them
                          # (exact, or the matmul pass with exact=False)
  .n_iter_                # = epochs
  .emeans_record_         # the run's record (below)

from mlx_kmeans import emeans_fit, emeans_reference
from mlx_kmeans import emeans          # emeans.batch_size(n, k, T), emeans.step0(n, k, T, B), emeans.plan(...)

C, record = emeans_fit(parts, k, epochs=3, seed=0, batch_size=None, step0=None,
                       init="random" | "k-means++" | array, assign="exact" | "approx",
                       method="auto", accumulate="auto")
C64, record = emeans_reference(X, k, epochs=3, seed=0, batch_size=None, step0=None,
                               init="random" | array, labels=None, revivals=None)
```

`emeans_fit` returns float32 centres, and `emeans_reference` returns float64 centres. The `record` holds:

- `B`, `a0`, `M` and `batches_per_epoch`;
- `dropped_per_epoch`: the rows each epoch leaves out;
- `seed`: drawn and recorded when you pass `None`, so any run can be repeated;
- `method` and `accumulate`: the batch assignment and accumulator that ran, plus `tiles1_fallback_rate` when that
  probe ran;
- `assign` and `approx_float16`: whether the approximate path ran, and whether its product was float16;
- `revived`: the clusters revived at each check;
- `train_mse`: one value per epoch, the mean over that epoch's batch rows of the float32 squared distance to the
  centre each row was assigned to, at the moment it was assigned;
- `last_epoch_empty`: clusters that received no rows in the last epoch;
- `timings`: `init_s`, `train_s`, `finish_s`, `total_s`, `permutation_first_wait_s` and `permutation_wait_s`.

`emeans_reference` runs the same algorithm in float64 NumPy, with the same initial rows, permutations and donor
draws. It rounds `X` and an `init` array to float32 first, so it computes on the same values `emeans_fit` does. Two
options make it a driven reference:
- `labels=`, a list of per-batch label arrays such as a GPU trace provides, replaces its own assignments;
- `revivals=`, one `(dead, donors)` pair per check, replaces its own revival decisions.

Driven by both, it checks the moving-average, revival and accumulation arithmetic, independently of float32
near-ties in the labels and of clusters within float32 rounding of the revival threshold. A starting point drawn by
`init="k-means++"` runs on the GPU and is kept in the record (`init_centres`) only when a trace is requested. To
replay such a run in the reference, pass `trace=` or give the starting centres as an array.

## What is checked, and what is measured

`tests/emeans.py` has three parts, and all three have run and pass. The NumPy and MLX-on-CPU parts (37 checks)
ran on 2026-11-13. The GPU part (34 checks) ran on 2026-11-14 on the M5 Max: 71/71 in all on the corrected `N`
check described under the GPU part, 70/71 on the check as first written. The same GPU session checked that exact
Lloyd is unchanged and then timed e-means against exact Lloyd and FAISS's Lloyd. Its record is
`benchmarks/NOTES.md` ("e-means, first measurements (2026-11-14)"), with one row per run in
`benchmarks/emeans_results.jsonl` (`L#` below is a line of that file), written by the harness `emeans_bench.py`.

**Lloyd is unchanged.** `core.py`'s three accumulation helpers became wrappers over new totals functions. On six
configurations, the lanes, blocks, sorted, one-hot, assignment and fit results that apply to each are bit-identical
to commit `9edfbe9` (HEAD). `tests/accuracy.py` (23/23), `tests/reuse.py` and `tests/large_k.py` pass.

**`--numpy` (NumPy only, 27 checks).**

- *Scaling laws.* The laws match hand-computed values, including the paraphrase's worked checks. Three examples:
  - n = 1e6, k = 16,384, T = 3 gives *B* = 65,536 and `a0` = 0.363.
  - k = 1,024, T = 25 gives *B* = 262,144 and a raw `a0` of 3.58, clipped to 1.
  - At n = 1e6, k = 4,096, log2 rounding gives 65,536 where linear rounding would give 32,768.
- *Schedule and random draws.* The cosine schedule, the donor draw (frequency proportional to `N`, drawn without
  replacement, lowest dead first) and the seeded batches behave as specified. The batches equal the sorted cuts of
  `default_rng([seed, 2, e]).permutation(n)`.
- *Reference self-checks.*
  - On overlapping blobs (30k × 32, k = 128), 3 epochs of the reference reach float64 Lloyd's loss to convergence
    from the same starting rows: the ratios were 0.9978, 0.9992 and 0.9976 for three seeds.
  - Its last-epoch mean equals the direct mean of its recorded assignments.
  - One full batch at `a0` = 1 equals one Lloyd step.
  - Driven by its own labels, and by its own labels and revival decisions, it reproduces itself. A revival decision
    handed in is the one applied.
  - With `a0` = 1, centres that no row reaches keep their position until the fifth batch revives them.

**`--mlx-cpu` (plain MLX ops on the CPU device, 10 more checks).**

- *GPU fit plumbing.* `emeans_fit` is driven by a batch step written in MLX ops, which exercises its gather, moving
  averages, revival, last-epoch two-sum accumulation, determinism and slicing. Five cases, one with k = 2,048 on
  8,192 rows (332 clusters revived in 4 checks, one cluster empty in the last epoch). It is checked at every batch:
  - labels: within accuracy.py's float32 window;
  - counts: exact;
  - one moving-average step from the fit's own state: within 3.0e-7 to 3.6e-7 of the RMS coordinate;
  - each revival decision: exactly the shared draw on the `N` the fit read.

  Against the reference driven by the same labels and revival decisions:
  - the trajectory: within 6.9e-7 to 8.8e-7;
  - the last-epoch means: within 1.1e-7 to 1.4e-7;
  - a centre kept from the moving average: bit-equal to the fit's last state, and within 2.1e-7 of the reference.

  The reference's own float64 revival decisions agreed at every check on these cases. The test reports them but
  does not judge them.
- *Segment layout.* The GPU-planned layout matches a NumPy emulation of the segment kernels.
- *Two-sum.* The MLX two-sum accumulation is exact to float64 on the test.

**GPU part (34 checks, run 2026-11-14).** It needs the Metal kernels. It covers:

- the same checks for every batch method and accumulator. Each case asserts the method and accumulator it ran. A
  `tiles1` chosen by auto is checked against its probe: it must have switched exactly when the fallback rate was over
  the limit.
- a case where revival fires, a case with `a0` = 1, a case over three slices, k-means++ initialisation, uncentred
  data (+1000 offset, the tiles1 probe), and the float16 and float32 approximate paths;
- a GPU that refuses the one-hot kernel (simulated): the fit falls back to rows and gives the same bits as
  `method="rows"`;
- bit-identical reruns, including when the data is sliced differently;
- `KMeans(algorithm="emeans")` end to end;
- the per-dataset caches: none keyed on batches, and the dataset's own entries not evicted.

On the GPU (`python tests/emeans.py`, 2026-11-14):

- Every case ran the batch method and accumulator it names. On the uncentred case the tiles1 probe measured a
  fallback rate of 1.000 and the fit switched to rows, as required.
- No exact case had a wrong label (at most 9 float32 ties per case). The approximate cases moved 1,305 labels
  (float16) and 370 (float32); that is printed, not judged, as the documented cost of `exact=False`.
- Counts were exact. One moving-average step from the fit's own state was within 2.2e-7 to 6.8e-7 of the RMS
  coordinate for the centres, and 1.0e-7 to 1.4e-7 for `N` (see the note below).
- Against the reference driven by the same labels and revival decisions: the trajectory was within 3.7e-7 to
  1.6e-6 (bound 1e-5; the highest on anisotropic d128 k256 with tiles1, tiles and pairs alike), the last-epoch means
  within 3.1e-8 to 1.6e-7, and every kept centre bit-equal to the fit's last state and within 5.8e-7 of the
  reference.
- The reference's own float64 revival decisions agreed with the fit's at every check of every case.
- Reruns were bit-identical with the same seed and with the data in 1 or 3 slices. A refused one-hot kernel fell
  back to rows with the same bits as `method="rows"`. `KMeans(algorithm="emeans")`, its `predict`, `exact=False` and
  its refusals passed, and so did the cache checks.

*The `N` check.* As first written, the moving-average check divided each cluster's `N` error by the mean `N`. On the
uncentred case one cluster took 645 of a 2,048-row batch (N = 538 against a mean of 8), and its float32 rounding,
0.72 ulp of its own N, read as 5.5e-6 against the 1e-6 bound. Across all 42 steps and 9,640 values, the fit's `N`
was bit-equal to a float32 emulation of `N (1 - a) + n a`, so the step is right and the scale was wrong. The check
now divides by `max(N_j, mean N)`. For a cluster at or below the mean that is the old number; above it, the check is
relative to `N_j`, the scale float32 rounds on, and looser than before by `N_j / mean N`. It still reads about 8e-4
for one misassigned row at that cluster, and the batch counts are checked exactly. That case also ended with 117 of
its 256 clusters empty in the last epoch after 556 revivals, with the same decisions in the float64 reference (see
choice 5).

**What is measured (2026-11-14).** One GPU session on AC power in High Power mode, with other processes running
beside it (archive copies, a download, a verification job: 80% to 542% of one core summed over other processes at
the run starts). Shapes: SIFT1M (1M × 128) at k = 1,024, 16,384 and 65,536, and GIST1M (1M × 960) at k = 1,024.
e-means ran at T = 1, 2, 3, 5 and 10 (1-5 at k = 16,384, 1-3 at k = 65,536) on the laws' *B* and `a0`, with three
seeds at k = 1,024, two at 16,384 and one at 65,536. Exact Lloyd (this library) ran from k-means++ and from
e-means' seed-0 rows, to a relative change of 1e-4 per step, with its per-step curve. FAISS's Lloyd (faiss-cpu
1.15.1, on the CPU) ran 25 iterations in its default setting and on all rows, except at k = 65,536. Every run
started with the per-dataset caches cleared, so every time is for a cold fit. Every method's centres are scored by
the library's exact pass; held-out MSE uses each dataset's query vectors (10,000 for SIFT, 1,000 for GIST).

| Shape | e-means T = 3: wall, loss / Lloyd (k-means++) | Exact Lloyd to tol: k-means++; random start | Rows |
|---|---|---|---|
| SIFT1M, k = 1,024 | 0.12-0.13 s, 0.99955-0.99964 | 0.83 s, 34 it; 0.75 s, 33 it | L7-L9, L16, L18 |
| GIST1M, k = 1,024 | 0.62-0.67 s, 0.9963-0.9964 | 2.44 s, 31 it; 1.96 s, 27 it | L28-L30, L37, L39 |
| SIFT1M, k = 16,384 | 1.53-1.54 s, 0.9991-0.9994 | 14.50 s, 26 it (7.33 s of it k-means++); 7.60 s, 27 it | L47-L48, L51, L53 |
| SIFT1M, k = 65,536 | 6.21 s, 0.9893 | 44.90 s, 18 it (28.02 s of it k-means++); 16.38 s, 17 it | L59-L61 |

- *Training loss against time.* On all four shapes e-means reached each of its training losses in less wall time
  than exact Lloyd took to reach the same loss, where Lloyd reached it. The exception is k = 65,536, T = 1:
  random-start Lloyd took 3.07 s and e-means 3.32 s (L57, L61). Where Lloyd did not reach the e-means loss (T ≥ 3 on
  SIFT; on GIST, T ≥ 2 against k-means++ Lloyd and T ≥ 5 against random-start Lloyd), e-means ended below the loss at
  which Lloyd stopped, in less time than that run. One such cell is level: GIST T = 10, 1.86-1.90 s against
  random-start Lloyd's 1.96 s, of which about 0.07 s is an artefact of run order (L34-L36, L39). Lloyd's first
  passes build its pruning state, which these cold times include.
- *Held-out error and recall do not follow everywhere.* On GIST, e-means' query MSE was above both Lloyd runs at
  every T: 1.0026 to 1.0343 times k-means++ Lloyd's, where random-start Lloyd was 0.9979 times (L22-L39); recall@10
  at nprobe 8 was 63.8% to 66.7%, against 68.5% (k-means++ Lloyd) and 65.6% (random start). On SIFT at k = 1,024 the
  held-out ratio stayed within 0.2% of the training ratio, and recall@10 at nprobe 8 was 84.7% to 85.0% at T ≥ 2,
  against 84.9% and 85.0%. At k = 16,384 and 65,536 query MSE was below k-means++ Lloyd's from T = 2 (0.9962-0.9975
  and 0.9870, L45-L46, L58; at T = 2 the training loss was still above it); at k = 65,536 recall@10 at nprobe 8 was
  53.6% and 54.2% (T = 2, 3) against 57.0% (k-means++ Lloyd) and 54.3% (random start), one seed.
- *The paper's claim.* The paper reports the epochs at which e-means' test MSE gets below Lloyd at 25 iterations,
  1.4 to 2.6 (its Table 4, against Flash-KMeans at 25 iterations; it normalises against Lloyd@25 as FAISS's default
  and uses 25 iterations of FAISS as the Lloyd reference on the CPU). On query MSE, against our Lloyd at 25
  iterations from e-means' seed-0 rows and FAISS's Lloyd on all rows at 25 iterations, only whole epochs were run, so
  each crossing is a bracket: between 1 and 2 epochs at SIFT k = 16,384 (both baselines) and k = 65,536 (one seed,
  against random-start Lloyd's stop at 17 iterations); between 2 and 3 at SIFT k = 1,024 against our Lloyd@25 and
  between 1 and 2 against FAISS's (the paper's Table 10 gives 2.0 epochs for SIFT at n/k = 1000); not reached by
  T = 10 on GIST1M (1.0045-1.0260 of our random-start Lloyd@25 at T = 10, L34-L36), which is not one of the
  paper's datasets. On training loss, against the same baselines, the brackets are 1-2 (SIFT k = 1,024), 3-5 (GIST)
  and 2-3 (SIFT k = 16,384 and 65,536).
- *Against the repo's earlier record* (benchmarks/NOTES.md, 2026-09-18): on SIFT1M k = 1,024 every e-means run, from
  T = 1 up, ended with a lower training loss than FAISS's subsampled default (5.0987e10, L20) and scikit-learn's
  MiniBatchKMeans in that record (5.1128e10), by the same scoring. The record's times were taken in another energy
  mode and are not compared.
- *Per batch.* An e-means batch costs about one Lloyd pass without pruning state over the same rows, and 1.3 to 4.5
  times a steady-state Lloyd pass: against the median step of k-means++ Lloyd, 1.30-2.65× on SIFT k = 1,024,
  3.83-4.51× on GIST, 1.87-2.67× at k = 16,384 and 2.36-3.75× at k = 65,536. The cascade (SIFT) and the per-centre
  bounds (GIST) that prune Lloyd's later passes do not apply to batches. The wall-time difference comes from reading
  the data T times instead of Lloyd's 17 to 34.
- *Initialisation.* e-means' random rows take 1 to 11 ms (seed 0). The library's k-means++ took 0.07, 0.34, 7.33 and
  28.02 s on the four shapes (L16, L37, L51, L60), most of k-means++ Lloyd's time at k = 16,384 and 65,536.
- *Revival.* Per run: 0-67 clusters at SIFT k = 1,024, 150-443 on GIST, 5,198-7,871 at k = 16,384, 28,273-37,879 at
  k = 65,536. Clusters empty in the last epoch appeared only at k = 65,536: 117, 17 and 7 at T = 1, 2, 3 (L57-L59).
- *T = 1* ended above Lloyd at 25 iterations on every shape with that row, by up to 3.8% (SIFT k = 16,384, L43);
  see choice 9.
- *Not measured:* scikit-learn's MiniBatchKMeans in this session (skipped; it is installed and can run next time),
  FAISS's e-means, fractional T, sizes other than 1M rows, Lloyd at 25 iterations at k = 65,536 (both runs met the
  tolerance first), a held-out time to match, and more than one seed at k = 65,536.

**Costs we know of.**

- *The host permutation.* Measured: the wait for epochs 1 on was 0.0000 to 0.0001 s on every run, so the worker
  thread's build finished inside each GPU epoch. The wait for epoch 0 (the part the initialisation did not cover)
  was 0 to 18 ms: 24% of a 0.062 s fit at SIFT k = 1,024, T = 1 (L1), 3.5% to 5.8% at T = 10, under 2% at
  k = 16,384 and 0.1% to 0.2% at 65,536. The earlier estimate that every epoch would wait was wrong. A permutation
  built on the GPU would remove at most the first wait.
- *Per-batch waits.* The host reads `N` every five batches and waits once per distance chunk when the batch's
  distance matrix spans more than one chunk (see "Where the host waits for the GPU").
- *The segment layout.* It sorts the labels with `mx.argsort` every batch. The library's counting sort
  (`_group_by_label`, for k ≤ 2,048) is faster than argsort at 2M-20M rows in the repo's measurements, but it reads
  its counts back to the host. A version that keeps them on the GPU is a follow-up. Whether this step is part of the
  per-batch cost above was not isolated.
- *Shapes without a fused path.* Where k·(dims + 2) ≤ 128 but the one-hot kernel does not apply (k not a multiple
  of 8, or dims % 8 = 7), Lloyd uses its plain fused kernel. e-means runs rows plus lanes there, which is correct but
  two passes instead of one.
- `blocks`, which auto never picks, is slow at large k × dims.

**Not in this version:**

- the slow-EMA finalizer;
- fractional epochs;
- IVF assignment for very large k (paper Sec. 3.4);
- reduced precision beyond the existing float16 matmul path.
