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

The paper leaves several details open. These are the choices this implementation makes.

1. **Batch counter.** *t* counts batches from 0 to M−1 across all epochs, and
   `a(t) = a0 (0.1 + 0.9 · 0.5 (1 + cos(π t / M)))`.
2. **Permutation and partial batch.** Each epoch draws a fresh seeded permutation. The last partial batch of each
   epoch is dropped, which matches `M = T floor(n/B)`. The `n − floor(n/B)·B` dropped rows differ from epoch to epoch.
   Because *B* is rounded to a power of two, the remainder can be large:
   - On the measured shapes (1M rows, k = 1,024 to 65,536, T = 1 to 5) it is 0.06% to 1.7% of each epoch, and
     8.2% at k = 1,024, T = 10 (B = 131,072, 7 batches; the rows' `dropped_per_epoch`).
   - Computed from the laws above, not run: at T = 25, 21.4% at 1M rows for k = 256 to 32,768 (B = 262,144 gives
     3 batches and a remainder of 213,568) and at 2M rows for k = 512 to 65,536 (B = 524,288, 3 batches).
   - Also computed, not run: at 1M rows, k = 65,536 and T = 25, the law gives B = 524,288: one batch per epoch,
     47.6% of the rows unused each epoch, so the last-epoch mean covers about half the data.

   The record's `dropped_per_epoch` and the verbose line give the count and fraction for a run. Whether to keep the
   partial batch instead is an open decision, not made here.
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
   from the random rows and contains every revival. The paper's experiments use T ≥ 2. On the benchmark shapes
   below, T = 1 ended above Lloyd at 25 iterations on every shape with that row (see "What is measured"). No stored
   run compares the last-epoch mean with the final moving-average centres at T = 1, so this page gives no number for
   that. Whether to refuse `epochs=1`, or to return the moving-average centres for it, is an open decision; the code
   does neither.

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

`tests/emeans.py` has three parts, and all three have run and pass. The counts and tolerances below are from those
runs' printed output, which is not stored in the repo (the tests are). The NumPy and MLX-on-CPU parts (37 checks)
ran on 2026-11-13. The GPU part (34 checks) ran on 2026-11-14 on the M5 Max: 71/71 in all on the corrected `N`
check described under the GPU part, 70/71 on the check as first written. The same GPU session checked that exact
Lloyd is unchanged and then timed e-means against exact Lloyd and FAISS's Lloyd. Its record is
`benchmarks/NOTES.md` ("e-means, first measurements (2026-11-14)"), with one row per run in
`benchmarks/emeans_results.jsonl` (`L#` below is a line of that file), written by the harness `emeans_bench.py`
(L1-L61). FAISS's default k-means was run again on 2026-11-15 and 2026-11-17, with five seeds at k = 1,024, two at
16,384 and one at 65,536 (L62-L74; the addendum "FAISS default k-means baseline (2026-11-15/17)" in the same notes),
by a short script that is not in the repo; the call and the scoring are given below.

**Lloyd is unchanged.** `core.py`'s three accumulation helpers became wrappers over new totals functions. On six
configurations, the lanes, blocks, sorted, one-hot, assignment and fit results that apply to each are bit-identical
to commit `9edfbe9`, the commit before the e-means work. `tests/accuracy.py` (23/23), `tests/reuse.py` and
`tests/large_k.py` pass.

**`--numpy` (NumPy only, 27 checks).**

- *Scaling laws.* The laws match hand-computed values (the table in the test). Three examples:
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
  coordinate for the centres, and 1.1e-7 to 1.4e-7 for `N` (see the note below).
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

**What is measured.** e-means against our exact Lloyd and FAISS's k-means on four shapes of 1M rows: SIFT1M
(1M × 128) at k = 1,024, 16,384 and 65,536, and GIST1M (1M × 960) at k = 1,024. e-means and our Lloyd ran on
2026-11-14 in one GPU session (L1-L61); FAISS's default k-means ran on the CPU on 2026-11-15 (L62-L73) and
2026-11-17 (L74).

*Read these first.*

- **What e-means gains here is training loss per unit of time.** At T = 3 its training loss was 0.04-1.1% below
  that of our Lloyd from k-means++ at its tolerance stop, in 1/3.7 to 1/9.5 of that run's time, seeding included.
- **Held-out error and recall are not better everywhere.** On GIST1M e-means' held-out MSE was above both of our
  Lloyd runs at every T, and its recall (seed 0) was below k-means++ Lloyd's at every T (63.83-66.65% against
  68.48%; random-start Lloyd had 65.59%). At SIFT k = 65,536 its recall was below k-means++ Lloyd's (53.546% and
  54.235% at T = 2 and 3, against 57.017%), and also below random-start Lloyd's 54.259% and FAISS's 54.473%, one run
  each.
- **Our Lloyd stopped at a relative change of 1e-4 per step and was still improving:** its last step lowered the
  loss by 7.6e-5 to 9.9e-5 relative, on every curve.
- **Seeds.** e-means ran three seeds at k = 1,024 (SIFT and GIST), two at 16,384 and one at 65,536. Our Lloyd ran
  once per start: k-means++ from seed 0, and the random start from e-means' seed-0 rows. e-means' recall is from
  seed 0 only. No difference here has a confidence interval.
- **Times.** e-means and our Lloyd ran on the GPU, FAISS on the CPU. The 2026-11-14 times were taken while other
  processes used 80% to 542% of one core and before the timing check was in use, so they are provisional.
- **Scope.** This e-means was written from the paper's text only (arXiv:2611.09501), without reading any e-means
  code. It is approximate and opt-in; exact Lloyd stays the default. FAISS's own e-means was not compared, and
  scikit-learn's MiniBatchKMeans was not run in this window.

*Conditions.*

- **Machine.** Apple M5 Max, 128 GB, macOS 27.2 (every row's `machine` field); AC power for every row (L1-L61's
  `power` field, the timing check's text in L62-L74). High Power mode (power mode 2): in the timing check's text for
  L62-L74; for L1-L61 it is from the session's run log at the start of each shape's run, which is not stored in the
  repo, as the rows record only the power source.
- **2026-11-14, rows stamped 15:16:40 to 15:40:55 (L1-L61):** e-means, our Lloyd, and FAISS's Lloyd in two settings (seed 0). Other
  processes ran beside the session (archive copies, a download, a verification job): 80% to 542% of one core summed
  at the run starts. No timing check was run (it was introduced the next day). These rows record commit
  `9edfbe9` with `dirty` false: the e-means code was then uncommitted work, committed afterwards as `11aa89f`, and the
  harness's dirty flag looks only at `kmeans.py` and `bench.py`.
- **2026-11-15 and 2026-11-17 (L62-L74):** FAISS's default k-means. A timing check (power source, power mode, busy
  processes) ran before each row and passed except two (L63 and L69, one busy process each), so those two times are
  provisional. Other processes used 45% to 151% of one core at the run starts. Battery 100% on AC, except L74 (79%,
  on AC). L74 (k = 65,536) ran on 2026-11-17 because the 2026-11-15 run at that k was stopped to free the machine.
- **Cold fits.** Every timed run started with the library's per-dataset caches cleared, after an untimed warm-up
  that compiles the kernels, so every e-means and Lloyd time is for a cold fit; each FAISS run built a new
  `faiss.Kmeans`.
- **e-means** ran at T = 1, 2, 3, 5 and 10 (1-5 at k = 16,384, 1-3 at k = 65,536) on the laws' *B* and `a0`, from
  random rows.
- **Our Lloyd** (exact) ran from k-means++ and from e-means' seed-0 rows to a relative change of 1e-4 per step,
  recording its per-step curve. "After 25 it" is the same run's centres after 25 iterations, timed to that point. At
  k = 65,536 both runs stopped before 25 iterations (18 and 17).
- **FAISS default k-means:** faiss-cpu 1.15.1 on the CPU, 18 threads, `faiss.Kmeans(d, k, niter=25, seed=s)` then
  `train(X)`, with FAISS's own limit of 256 training rows per centre. That is a 262,144-row sample at k = 1,024 and
  all 1,000,000 rows at k = 16,384 and 65,536 (at k = 65,536 FAISS warned that it wants at least 2,555,904 points,
  its minimum of 39 per centre). L62-L74 record the version, threads and rows used. **FAISS on all rows**
  (2026-11-14, seed 0) is the same call with the limit lifted; the 2026-11-14 FAISS rows (L20, L21, L41, L42, L55,
  L56) do not record the FAISS version or thread count.
- **Time** is wall time to a finished set of centres: e-means including its random-row start and its last-epoch mean,
  our Lloyd including its seeding, FAISS from creating the `faiss.Kmeans` object through `train`. The scoring pass is
  in no time.
- **Scoring**, the same for every method. Training loss is the sum of squared distances over all 1M rows by the
  library's exact pass. Held-out MSE is the mean squared distance from the dataset's query vectors (10,000 for SIFT,
  1,000 for GIST) to their nearest centre. Recall@10 is the share of each query's 10 true nearest neighbours that a
  FAISS IVF-Flat index built from the centres returns in its top 10 at nprobe 8; it is printed exactly (SIFT has
  100,000 neighbours to find, GIST 10,000). In the tables both losses are ratios to our Lloyd from k-means++ at its
  tolerance stop on the same shape (the row marked 1, with its absolute value).

*SIFT1M (1M × 128), k = 1,024*

| Method | Seeds | Time, s | Training loss ratio | Held-out MSE ratio | Recall@10 at nprobe 8, % (e-means: seed 0) | Rows |
|---|---|---|---|---|---|---|
| e-means, T = 1 | 0-2 | 0.06-0.07 | 1.00771-1.00824 | 1.00471-1.00498 | 84.057 | L1-L3 |
| e-means, T = 2 | 0-2 | 0.09-0.10 | 1.00131-1.00142 | 0.99947-1.00065 | 84.730 | L4-L6 |
| e-means, T = 3 | 0-2 | 0.12-0.13 | 0.99955-0.99964 | 0.99823-0.99935 | 84.938 | L7-L9 |
| e-means, T = 5 | 0-2 | 0.17-0.19 | 0.99877-0.99921 | 0.99808-0.99903 | 84.842 | L10-L12 |
| e-means, T = 10 | 0-2 | 0.27-0.29 | 0.99884-0.99943 | 0.99771-0.99901 | 84.963 | L13-L15 |
| Our Lloyd, k-means++ start, to tol (34 it) | k-means++ seed 0 | 0.83 | 1 (= 5.0474e10) | 1 (= 50,750.4) | 84.894 | L16 |
| Our Lloyd, k-means++ start, after 25 it (same run) | k-means++ seed 0 | 0.64 | 1.00101 | 1.00112 | 84.675 | L17 |
| Our Lloyd, random start, to tol (33 it) | e-means' seed-0 rows | 0.75 | 1.00048 | 0.99920 | 85.012 | L18 |
| Our Lloyd, random start, after 25 it (same run) | e-means' seed-0 rows | 0.58 | 1.00146 | 0.99997 | 84.989 | L19 |
| FAISS default k-means (CPU), 25 it, 262,144 rows used | 0-4 | 4.93-5.14 | 1.00936-1.01067 | 1.00594-1.00832 | 84.009-84.215 | L62-L66 |
| FAISS k-means on all rows (CPU), 25 it, 2026-11-14 window | 0 | 18.87 | 1.00167 | 1.00150 | 84.447 | L21 |

FAISS's default in the 2026-11-14 window (L20, seed 0) gave the same training loss, held-out MSE and recall as L62,
in 5.07 s.

*GIST1M (1M × 960), k = 1,024*

| Method | Seeds | Time, s | Training loss ratio | Held-out MSE ratio | Recall@10 at nprobe 8, % (e-means: seed 0) | Rows |
|---|---|---|---|---|---|---|
| e-means, T = 1 | 0-2 | 0.24-0.29 | 1.00282-1.00298 | 1.02505-1.03425 | 63.83 | L22-L24 |
| e-means, T = 2 | 0-2 | 0.42-0.48 | 0.99754-0.99773 | 1.01133-1.02451 | 65.17 | L25-L27 |
| e-means, T = 3 | 0-2 | 0.62-0.67 | 0.99628-0.99640 | 1.00875-1.02062 | 66.20 | L28-L30 |
| e-means, T = 5 | 0-2 | 0.97-1.04 | 0.99527-0.99539 | 1.00851-1.02077 | 65.77 | L31-L33 |
| e-means, T = 10 | 0-2 | 1.86-1.90 | 0.99478-0.99503 | 1.00260-1.02400 | 66.65 | L34-L36 |
| Our Lloyd, k-means++ start, to tol (31 it) | k-means++ seed 0 | 2.44 | 1 (= 1.1124e6) | 1 (= 1.0997) | 68.48 | L37 |
| Our Lloyd, k-means++ start, after 25 it (same run) | k-means++ seed 0 | 2.14 | 1.00062 | 1.00009 | 68.77 | L38 |
| Our Lloyd, random start, to tol (27 it) | e-means' seed-0 rows | 1.96 | 0.99610 | 0.99793 | 65.59 | L39 |
| Our Lloyd, random start, after 25 it (same run) | e-means' seed-0 rows | 1.86 | 0.99628 | 0.99810 | 65.42 | L40 |
| FAISS default k-means (CPU), 25 it, 262,144 rows used | 0-4 | 11.04-11.26 | 1.00523-1.00558 | 1.00593-1.02406 | 65.93-66.58 | L67-L71 |
| FAISS k-means on all rows (CPU), 25 it, 2026-11-14 window | 0 | 42.82 | 0.99600 | 0.99821 | 65.83 | L42 |

FAISS's default in the 2026-11-14 window (L41, seed 0) gave the same scores as L67, in 12.11 s.

*SIFT1M (1M × 128), k = 16,384*

| Method | Seeds | Time, s | Training loss ratio | Held-out MSE ratio | Recall@10 at nprobe 8, % (e-means: seed 0) | Rows |
|---|---|---|---|---|---|---|
| e-means, T = 1 | 0-1 | 0.59-0.70 | 1.03771-1.03814 | 1.00413-1.00860 | 58.782 | L43-L44 |
| e-means, T = 2 | 0-1 | 1.14-1.24 | 1.00611-1.00617 | 0.99622-0.99751 | 60.643 | L45-L46 |
| e-means, T = 3 | 0-1 | 1.53-1.54 | 0.99914-0.99940 | 0.99559-0.99636 | 61.300 | L47-L48 |
| e-means, T = 5 | 0-1 | 2.41-2.63 | 0.99169-0.99187 | 0.99447-0.99579 | 62.128 | L49-L50 |
| Our Lloyd, k-means++ start, to tol (26 it) | k-means++ seed 0 | 14.50 | 1 (= 3.8191e10) | 1 (= 41,798.4) | 61.978 | L51 |
| Our Lloyd, k-means++ start, after 25 it (same run) | k-means++ seed 0 | 14.24 | 1.00008 | 1.00000 | 61.974 | L52 |
| Our Lloyd, random start, to tol (27 it) | e-means' seed-0 rows | 7.60 | 1.00146 | 0.99822 | 61.409 | L53 |
| Our Lloyd, random start, after 25 it (same run) | e-means' seed-0 rows | 7.09 | 1.00164 | 0.99831 | 61.381 | L54 |
| FAISS default k-means (CPU), 25 it, 1,000,000 rows used | 0-1 | 295.63-296.54 | 1.00169 | 0.99647-0.99775 | 61.415-61.475 | L72-L73 |

FAISS's default uses all rows at this k (256 × k > n). The 2026-11-14 window's FAISS rows (seed 0: L55 default, L56
all rows) gave the same scores as L72, in 294.19 s and 294.52 s.

*SIFT1M (1M × 128), k = 65,536*

| Method | Seeds | Time, s | Training loss ratio | Held-out MSE ratio | Recall@10 at nprobe 8, % (e-means: seed 0) | Rows |
|---|---|---|---|---|---|---|
| e-means, T = 1 | 0 | 3.32 | 1.07297 | 0.99839 | 50.343 | L57 |
| e-means, T = 2 | 0 | 4.41 | 1.00533 | 0.98702 | 53.546 | L58 |
| e-means, T = 3 | 0 | 6.21 | 0.98928 | 0.98691 | 54.235 | L59 |
| Our Lloyd, k-means++ start, to tol (18 it) | k-means++ seed 0 | 44.90 | 1 (= 3.0551e10) | 1 (= 39,667.2) | 57.017 | L60 |
| Our Lloyd, random start, to tol (17 it) | e-means' seed-0 rows | 16.38 | 1.00303 | 0.99117 | 54.259 | L61 |
| FAISS default k-means (CPU), 25 it, 1,000,000 rows used | 0 | 1,181.42 | 1.00249 | 0.98876 | 54.473 | L74 |

One run per method at this k. No Lloyd row at 25 iterations: both runs stopped first. The FAISS row ran on
2026-11-17 (battery 79% on AC).

- *Training loss against time.* On all four shapes e-means reached each of its training losses in less wall time
  than exact Lloyd took to reach the same loss, where Lloyd reached it. The exception is k = 65,536, T = 1:
  random-start Lloyd took 3.07 s and e-means 3.32 s (L57, L61). Where Lloyd did not reach the e-means loss (T ≥ 3 on
  SIFT; on GIST, T ≥ 2 against k-means++ Lloyd and T ≥ 5 against random-start Lloyd), e-means ended below the loss at
  which Lloyd stopped, in less time than that run. One such cell is level: GIST T = 10, 1.86-1.90 s against
  random-start Lloyd's 1.96 s (L34-L36, L39), of which about 0.07 s is that run's slower start: its random-row gather
  took 73 ms (see the run-order note in `benchmarks/NOTES.md`; the cause is untested). Lloyd's first passes build
  its pruning state, which these cold times include. At T = 3, e-means took 1/3.7 to 1/9.5 of the time of k-means++
  Lloyd to its tolerance stop, seeding included: Lloyd's time over e-means' was 3.66-3.96 on GIST, 6.26-6.74 at SIFT
  k = 1,024, 7.23 at k = 65,536 and 9.42-9.45 at k = 16,384 (L7-L9, L16, L28-L30, L37, L47-L48, L51, L59, L60).
- *Held-out error and recall do not follow everywhere.* On GIST, e-means' query MSE was above both Lloyd runs at
  every T: 1.0026 to 1.0342 times k-means++ Lloyd's, where random-start Lloyd was 0.9979 times (L22-L39); recall@10
  at nprobe 8 (seed 0) was 63.83% to 66.65%, against 68.48% (k-means++ Lloyd) and 65.59% (random start). On SIFT
  at k = 1,024, at T ≥ 2, the held-out ratio stayed within 0.2% of the training ratio (at T = 1 it was 0.27% to
  0.33% below it, L1-L3), and recall@10 at nprobe 8 was 84.7% to 85.0%, against 84.9% and 85.0%. Query MSE was
  below k-means++ Lloyd's from T = 2 at k = 16,384 (0.9962-0.9975 at T = 2, L45-L46; 1.0041-1.0086 at T = 1) and
  from T = 1 at k = 65,536 (0.9984 at T = 1, 0.9870 at T = 2, L57-L58); at T = 2 the training loss was still above
  k-means++ Lloyd's at both k (1.0061-1.0062 and 1.0053). At k = 65,536 recall@10 at nprobe 8 was
  53.5% and 54.2% (T = 2, 3) against 57.0% (k-means++ Lloyd), 54.3% (random start) and 54.5% (FAISS's default,
  L74), one seed.
- *The paper's claim.* The paper reports the epochs at which e-means' test MSE gets below Lloyd at 25 iterations,
  1.4 to 2.6 (its Table 4, against Flash-KMeans at 25 iterations; it normalises against Lloyd@25 as FAISS's default
  and uses 25 iterations of FAISS as the Lloyd reference on the CPU). On query MSE, against our Lloyd at 25
  iterations from e-means' seed-0 rows and FAISS's Lloyd on all rows at 25 iterations, only whole epochs were run, so
  each crossing is a bracket: between 1 and 2 epochs at SIFT k = 16,384 (both baselines) and k = 65,536 (one seed,
  against random-start Lloyd's stop at 17 iterations); between 2 and 3 at SIFT k = 1,024 against our Lloyd@25 on
  the median of three seeds (1.00013 at T = 2; seed 0 was already below, 0.9995, L4) and between 1 and 2 against
  FAISS's for every seed (the paper's Table 10 gives 2.0 epochs for SIFT at n/k = 1000); not reached by
  T = 10 on GIST1M (1.0045-1.0259 of our random-start Lloyd@25 at T = 10, L34-L36), which is not one of the
  paper's datasets. On training loss, against the same baselines, the brackets are 1-2 (SIFT k = 1,024), 3-5 (GIST:
  for every seed against FAISS's, 1.0003-1.0004 at T = 3; on the seed median against our Lloyd@25, where seed 1
  was already 3e-6 below at T = 3, L29) and 2-3 (SIFT k = 16,384 and 65,536).
- *Against the repo's earlier record* (benchmarks/NOTES.md, 2026-09-18): on SIFT1M k = 1,024 every e-means run, from
  T = 1 up, ended with a lower training loss than FAISS's subsampled default (5.0987e10, L20) and scikit-learn's
  MiniBatchKMeans in that record (5.1128e10), by the same scoring. The record's times were taken in another energy
  mode and are not compared.
- *Per batch.* Compared per row: e-means' training time less both host permutation waits (`timings.train_s` minus
  `permutation_first_wait_s` and `permutation_wait_s`), divided by its M × B batch rows, against a k-means++ Lloyd
  step divided by the 1M rows. On that basis an e-means batch costs about one Lloyd pass without pruning state over
  the same rows (against Lloyd's first step: 1.04-2.11× on SIFT k = 1,024, 1.00-1.17× on GIST, 0.98-1.40× at
  k = 16,384, 1.21-1.92× at k = 65,536), and 1.3 to 4.5 times a steady-state Lloyd pass: against the median step of
  k-means++ Lloyd, 1.30-2.65× on SIFT k = 1,024, 3.83-4.51× on GIST, 1.87-2.67× at k = 16,384 and 2.36-3.75× at
  k = 65,536. The cascade (SIFT) and the per-centre
  bounds (GIST) that prune Lloyd's later passes do not apply to batches. Lloyd ran 17 to 34 iterations to its stop
  where e-means read the data T times. Lloyd's time also includes its seeding: k-means++ took 9.0%, 13.8%, 50.6% and
  62.4% of k-means++ Lloyd's time (SIFT k = 1,024, GIST, SIFT k = 16,384 and 65,536). How the wall-time difference
  splits among the reads, the seeding and the per-batch cost was not measured.
- *Initialisation.* e-means' random rows take 1 to 11 ms (seed 0). The library's k-means++ took 0.07, 0.34, 7.33 and
  28.02 s on the four shapes (L16, L37, L51, L60), most of k-means++ Lloyd's time at k = 16,384 and 65,536.
- *Revival.* Per run: 0-67 clusters at SIFT k = 1,024, 150-443 on GIST, 5,198-7,871 at k = 16,384, 28,273-37,879 at
  k = 65,536. Clusters empty in the last epoch appeared only at k = 65,536: 117, 17 and 7 at T = 1, 2, 3 (L57-L59).
- *T = 1* ended above Lloyd at 25 iterations on every shape with that row, by up to 3.8% (SIFT k = 16,384, L43
  against k-means++ Lloyd's L52); see choice 9.
- *Not measured:* scikit-learn's MiniBatchKMeans in this window (skipped; it is installed and can run next time),
  FAISS's own e-means, e-means and Lloyd times under the timing check, fractional T, sizes other than 1M rows, Lloyd
  at 25 iterations at k = 65,536 (both runs met the tolerance first), a held-out time to match, and more than one
  run per method at k = 65,536.

**Costs we know of.**

- *The host permutation.* Measured: the wait for epochs 1 on was 0.0000 to 0.0001 s on every run, so the worker
  thread's build finished inside each GPU epoch. The wait for epoch 0 (the part the initialisation did not cover)
  was 0 to 18 ms: 24% of a 0.062 s fit at SIFT k = 1,024, T = 1 (L1), 3.4% to 5.8% at SIFT k = 1,024, T = 10
  (under 1% on GIST at T = 10), under 2% at k = 16,384 and 0.1% to 0.2% at 65,536, as shares of e-means' own
  training time. The earlier estimate that every epoch would wait was wrong. A permutation
  built on the GPU would remove at most the first wait.
- *Per-batch waits.* The host reads `N` every five batches and waits once per distance chunk when the batch's
  distance matrix spans more than one chunk (see "Where the host waits for the GPU").
- *The segment layout.* It sorts the labels with `mx.argsort` every batch. The library's counting sort
  (`_group_by_label`, for k ≤ 2,048) is an alternative, but it reads its counts back to the host; how it compares
  with the argsort here has not been measured. A version that keeps the counts on the GPU is a follow-up. Whether
  this step is part of the per-batch cost above was not isolated.
- *Shapes without a fused path.* Where k·(dims + 2) ≤ 128 but the one-hot kernel does not apply (k not a multiple
  of 8, or dims % 8 = 7), Lloyd uses its plain fused kernel. e-means runs rows plus lanes there, which is correct but
  two passes instead of one.
- `blocks`, which auto never picks, is slow at large k × dims.

**Not in this version:**

- the slow-EMA finalizer;
- fractional epochs;
- IVF assignment for very large k (paper Sec. 3.4);
- reduced precision beyond the existing float16 matmul path.
