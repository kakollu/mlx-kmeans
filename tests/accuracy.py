#!/usr/bin/env python3
"""Accuracy suite: every assignment-pass path must match float64 ground truth.

For each case and each path (rows, pairs):
  1. Labels: every point's center is optimal in float64, up to float32 resolution:
         d64(x, label) <= d64(x, best) + 8 * eps32 * (|x|^2 + |c|^2)
     (points whose float64 distances to two centers differ by less than that are ties at float32 precision).
     Strict disagreements with float64 are reported too.
  2. Accumulation: counts exact; every new center (sum / count) within 1e-6 of the float64 mean over the same
     labels, measured relative to the data's RMS coordinate (~8x float32 eps: float32 centers can't do much
     better); inertia within 1e-6 relative.

Run:  python3 tests/accuracy.py        (exit code 1 on any failure)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kmeans as km  # noqa: E402

EPS32 = float(np.finfo(np.float32).eps)


def reference(X, C, chunk_terms=20_000_000):
    """Exact float64 nearest center (lowest index on exact ties) and its distance, for every row."""
    X64, C64 = X.astype(np.float64), C.astype(np.float64)
    csq = (C64 ** 2).sum(1)
    n, k = len(X), len(C)
    best_d = np.empty(n)
    best_c = np.empty(n, np.int64)
    step = max(1, chunk_terms // (k * X.shape[1]))
    for s in range(0, n, step):
        x = X64[s:s + step]
        approx = csq[None, :] - 2 * x @ C64.T
        # float64 expanded-form error is ~1e-15 relative; take every center within a generous window, then exact
        lo = approx.min(1, keepdims=True)
        win = 1e-9 * ((x ** 2).sum(1, keepdims=True) + csq[None, :]) + 1e-300
        cand = approx <= lo + 2 * win
        d = np.full(approx.shape, np.inf)
        rows, cols = np.nonzero(cand)
        d[rows, cols] = ((x[rows] - C64[cols]) ** 2).sum(1)
        best_c[s:s + step] = d.argmin(1)
        best_d[s:s + step] = d[np.arange(len(x)), best_c[s:s + step]]
    return best_c, best_d


def check(name, X, C, slice_rows=None, dist_bytes=None):
    km.SLICE_ROWS = slice_rows or 100_000_000
    if dist_bytes:
        km.DIST_BYTES = dist_bytes
    mx = km._mx()
    per = km.slice_rows(X.shape[1])
    parts = [mx.array(X[s:s + per]) for s in range(0, len(X), per)]
    ref_c, ref_d = reference(X, C)
    X64, C64 = X.astype(np.float64), C.astype(np.float64)
    ok = True
    for method in ["rows", "pairs"]:
        sums, counts, inertia, labels = km.assign_mlx(parts, C, method=method, return_labels=True)
        d_lab = ((X64 - C64[labels]) ** 2).sum(1)
        tol = 8 * EPS32 * ((X64 ** 2).sum(1) + (C64[labels] ** 2).sum(1))
        bad = int((d_lab > ref_d + tol).sum())
        strict = int((labels != ref_c).sum())
        k, d = C.shape
        ref_counts = np.bincount(labels, minlength=k)
        ref_sums = np.zeros((k, d))
        np.add.at(ref_sums, labels, X64)
        ref_inertia = d_lab.sum()
        used = ref_counts > 0
        scale = float(np.sqrt((X64 ** 2).mean()))
        center_err = float(np.max(np.abs(sums[used] - ref_sums[used]) / ref_counts[used, None])) / scale
        inertia_err = abs(inertia - ref_inertia) / max(ref_inertia, 1e-30)
        passed = bad == 0 and np.array_equal(counts, ref_counts) and center_err <= 1e-6 and inertia_err <= 1e-6
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {name:44s} {method:8s}  wrong labels {bad}  "
              f"(float32-level ties {strict - bad})  counts {'ok' if np.array_equal(counts, ref_counts) else 'MISMATCH'}  "
              f"center err {center_err:.1e}  inertia err {inertia_err:.1e}")
    km.SLICE_ROWS, km.DIST_BYTES = 100_000_000, 512 << 20
    return ok


def blobs(rng, n, d, true_k, spread, noise):
    centers = rng.uniform(-spread, spread, size=(true_k, d))
    return (centers[rng.integers(0, true_k, n)] + rng.normal(0, noise, (n, d))).astype(np.float32)


def main():
    rng = np.random.default_rng(0)
    results = []
    X = blobs(rng, 300_000, 8, 16, 10, 1)
    results.append(check("separated d8 k16", X, X[rng.choice(len(X), 16, replace=False)]))
    X = blobs(rng, 300_000, 32, 32, 1, 1)  # noise as large as the center spacing: many near-ties
    results.append(check("overlapping d32 k256", X, X[rng.choice(len(X), 256, replace=False)]))
    X = blobs(rng, 200_000, 16, 4, 10, 1)
    results.append(check("k >> true clusters d16 k512", X, X[rng.choice(len(X), 512, replace=False)]))
    X = blobs(rng, 100_000, 8, 8, 10, 1)
    C = X[rng.choice(len(X), 32, replace=False)]
    C[16:] = C[:16]  # exact duplicate centers: ties must go to the lower index
    results.append(check("duplicate centers d8 k32", X, C))
    X = np.repeat(blobs(rng, 5_000, 8, 8, 10, 1), 20, axis=0)  # every point repeated 20x
    results.append(check("duplicate points d8 k64", X, X[rng.choice(len(X), 64, replace=False)]))
    X = blobs(rng, 7, 3, 2, 10, 1)
    results.append(check("tiny n=7 d3 k5", X, X[:5].copy()))
    X = (blobs(rng, 100_000, 8, 16, 10, 1) + 1000).astype(np.float32)  # large offset stresses float32 cancellation
    results.append(check("offset +1000 d8 k64", X, X[rng.choice(len(X), 64, replace=False)]))
    X = rng.random((60_000, 128), dtype=np.float32) * 255  # SIFT-like range
    results.append(check("uniform 0-255 d128 k1000", X, X[rng.choice(len(X), 1000, replace=False)]))
    X = blobs(rng, 20_000, 960, 50, 1, 1)
    results.append(check("high dims d960 k300", X, X[rng.choice(len(X), 300, replace=False)]))
    X = blobs(rng, 250_003, 16, 16, 5, 1)
    results.append(check("many slices + chunks d16 k100", X, X[rng.choice(len(X), 100, replace=False)],
                         slice_rows=60_001, dist_bytes=4 * 100 * 7_777))
    # accumulation stress: ~5M points per cluster at |x| ~ 1000, one slice, few large blocks
    X = (blobs(rng, 20_000_000, 2, 4, 10, 1) + 1000).astype(np.float32)
    results.append(check("accumulation 20M rows +1000 d2 k4", X, X[rng.choice(len(X), 4, replace=False)]))
    print(f"\n{sum(results)}/{len(results)} cases passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
