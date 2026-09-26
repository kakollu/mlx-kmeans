# mlx-kmeans — project state

Last updated: 2026-09-26

## Where it stands

Exact, deterministic k-means on Apple Silicon GPUs via MLX and custom Metal kernels. `main` on
`kakollu/mlx-kmeans` is at the September 25 push (`a51fe01`); commits after it are local (see `git log
origin/main..HEAD`). Work on this repo pauses after 2026-09-25 by decision.

Suite (M5 Max, 128 GB, AC): 3.5-22x per pass over the quickest valid time of the compared libraries on six
shapes; floors and ratios in `BENCHMARKS.md`. GIST 1M k=1024: 176 ms per plain pass, 46-49 ms per
converged-regime iteration on the bounds path, 20 iterations 3.51 -> 1.41 s. The no-admin benchmark
(`./no_admin_bench.sh`, from a fresh download) reproduces the suite numbers.

## What exists (September 24-25 work, all in NOTES with numbers)

- Exact assignment kernels: rows, pairs, tiles, tiles1, the cascade (128-256 dims), the one-hot fused kernel
  (k <= 128, dims <= 96, `method="onehot"`), and the per-centre bounds path between iterations (256+ dims,
  float16 bounds, float16 screen on visits, built only when a sampled prediction says it pays).
- Deterministic accumulation (counting sort + segment sums, Kahan); `accumulate="atomic"` opt-in.
- `exact=False`: matmul-based approximate assignment (float16 where the data's norms allow, float32 otherwise).
- Reuse safety: per-dataset caches keyed on live references and released with the data; input checks raise
  instead of aborting; `tests/reuse.py`.

## Open, with the measured reason each is where it is

- Low-dimensional shapes are 2.7-7x from the one-read floor (single-cell 5.4x, geo-trips 7.1x). Three
  designs meet at 4.2 ms on single-cell: an exact, deterministic pass costs the assignment's arithmetic twice.
  What would move it: an assignment at the matrix unit's ceiling (tiles1 is at ~40%) and an accumulation
  that does not repeat it - which MACHINE-PROFILE's occupancy table rules out in threadgroup memory above
  k x (dims+2) of a few hundred floats.
- The converged high-dimensional regime is 46-49 ms against a ~37 ms bound. 99.7% of its visits refresh bounds
  that decayed rather than test close centres (NOTES, "Bound freshness, measured"); the largest movers get
  only 7-15% of visits, and the drift matrix is only moderately low-rank (r=32: 70-85% energy), so a
  low-rank correction is worth ~1.2x for an SVD in the loop. A different bound family is a research
  question, not a step.
- `mx.matmul` is not float32 arithmetic (8e-4 |x||c| error); no exact path can be built on it. Closed.
- No batched interface for many small problems (PQ codebooks); no fit-to-tolerance config in the suite;
  small data (< a few hundred thousand rows) pays the 0.18 ms round trip per iteration.
- `exact=False` float16 guard bounds the product's range, not its precision (audit finding): uncentred
  data with |x||c| ~ 1e4 passes and gets wrong labels; data below ~1e-5 collapses to subnormals.

## How to resume

Read `README.md`, then `benchmarks/NOTES.md` from "Optimal at 960 dims" onward, then `possible.py`. Run both
test suites first. Numbers are only comparable on AC power with heavy apps closed; the first measurement
after creating a dataset runs 1.5-2x slow (warm up, then time).
