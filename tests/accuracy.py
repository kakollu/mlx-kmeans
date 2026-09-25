#!/usr/bin/env python3
"""Accuracy suite: every assignment-pass path must match float64 ground truth.

For each case and each combination of nearest-center path (rows, pairs, tiles, tiles1 where dims and k
allow)
and accumulation (blocks, sorted, lanes where k*(dims+2) allows):
  1. Labels: every point's center is optimal in float64, up to float32 resolution:
         d64(x, label) <= d64(x, best) + 8 * eps32 * (|x|^2 + |c|^2)
     (points whose float64 distances to two centers differ by less than that are ties at float32 precision).
     Strict disagreements with float64 are reported too.
  2. Accumulation: counts exact; every new center (sum / count) within 1e-6 of the float64 mean over the same
     labels, measured relative to the data's RMS coordinate (~8x float32 eps: float32 centers can't do much
     better); inertia within 1e-6 relative.

Plus full Lloyd iterations with empty-cluster relocation, checked against scikit-learn's lloyd from the same start.

Run:  python3 tests/accuracy.py        (exit code 1 on any failure)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx_kmeans as km  # noqa: E402
from mlx_kmeans import core  # noqa: E402

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
    km.core.SLICE_ROWS = slice_rows or 100_000_000
    if dist_bytes:
        km.core.DIST_BYTES = dist_bytes
    mx = km._mx()
    per = km.slice_rows(X.shape[1])
    parts = [mx.array(X[s:s + per]) for s in range(0, len(X), per)]
    if slice_rows:
        assert per == slice_rows and len(parts) == (len(X) + per - 1) // per
    if dist_bytes:
        assert km.core.DIST_BYTES == dist_bytes
    ref_c, ref_d = reference(X, C)
    X64, C64 = X.astype(np.float64), C.astype(np.float64)
    ok = True
    combos = [("rows", "blocks"), ("rows", "sorted"), ("pairs", "blocks"), ("pairs", "sorted")]
    if km.core._onehot_plan(min(p.shape[0] for p in parts), X.shape[1], len(C)):
        combos += [("onehot", "auto")]
    if 2 * len(C) * (C.shape[1] + 2) <= km.core.TG_FLOATS:
        combos += [("rows", "lanes")]
    if len(C) * (C.shape[1] + 2) <= km.core.ATOMIC_MAX_KW:
        combos += [("rows", "atomic")]
    if C.shape[1] % 8 == 0 and len(C) % 8 == 0:
        combos += [("tiles", "blocks"), ("tiles", "sorted")]
    if km.core._tiles1_ok(C.shape[1], len(C)):
        combos += [("tiles1", "blocks"), ("tiles1", "sorted")]
    if km.core._cascade_ok(C.shape[1], len(C)):
        km.core._cascade_state.clear()
        combos += [("cascade", "blocks")]
    for method, accumulate in combos:
        sums, counts, inertia, labels = km.assign_mlx(parts, C, method=method, return_labels=True, accumulate=accumulate)
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
        print(f"{'PASS' if passed else 'FAIL'}  {name:36s} {method + '/' + accumulate:13s}  wrong labels {bad}  "
              f"(float32-level ties {strict - bad})  counts {'ok' if np.array_equal(counts, ref_counts) else 'MISMATCH'}  "
              f"center err {center_err:.1e}  inertia err {inertia_err:.1e}")
    km.core.SLICE_ROWS, km.core.DIST_BYTES = 100_000_000, 512 << 20
    return ok


def reference_lloyd_step(X, C, labels):
    """Independent float64 Lloyd step (accumulation + empty-cluster rules of km.lloyd_step) for given labels."""
    X64, C64 = X.astype(np.float64), C.astype(np.float64)
    best = ((X64 - C64[labels]) ** 2).sum(1)
    k = len(C)
    counts = np.bincount(labels, minlength=k)
    sums = np.zeros((k, X.shape[1]))
    np.add.at(sums, labels, X64)
    empty = np.flatnonzero(counts == 0)
    if len(empty) and best.max() > 0:
        order = sorted(range(len(X)), key=lambda i: (-best[i], i))[:len(empty)] if len(X) < 5000 else \
            [i for _, i in sorted((-best[i], i) for i in np.argsort(-best, kind="stable")[:len(empty) + 50])][:len(empty)]
        for new, f in zip(empty, order):
            sums[labels[f]] -= X64[f]
            counts[labels[f]] -= 1
            sums[new] = X64[f]
            counts[new] = 1
    out = np.empty_like(C64)
    has = counts > 0
    out[has] = sums[has] / counts[has, None]
    out[~has] = out[np.argmax(counts)]
    return out, len(empty)


def check_relocation(name, X, C0, steps=8, vs_sklearn=False):
    """Lloyd steps including empty-cluster relocation, each step started from the same centers.

    Always: our labels are float64-optimal up to float32 resolution, and given those labels, our centers match the
    float64 reference step above within 1e-6 of the data scale (labels are shared so float32-level ties, which either
    implementation may break either way, can't masquerade as relocation differences).
    With vs_sklearn (cases where at most one cluster empties per step, so pairing order can't matter): steps that
    relocate must also match scikit-learn's lloyd within 1e-5 (scikit-learn sums centers in float32) or, if a
    float32-level tie label differs, give inertia within 1e-6.
    """
    from sklearn.cluster import KMeans
    mx = km._mx()
    parts = [mx.array(X)]
    scale = float(np.sqrt((X.astype(np.float64) ** 2).mean()))
    C, empties, ref_err, sk_err = C0.copy(), 0, 0.0, 0.0
    for _ in range(steps):
        ours, _, n_empty = km.lloyd_step(parts, C)
        labels = km.assign_mlx(parts, C, return_labels=True)[3]
        _, ref_best = reference(X, C)
        X64, C64 = X.astype(np.float64), C.astype(np.float64)
        d_lab = ((X64 - C64[labels]) ** 2).sum(1)
        if (d_lab > ref_best + 8 * EPS32 * ((X64 ** 2).sum(1) + (C64[labels] ** 2).sum(1))).any():
            ref_err = float("inf")  # a label that isn't optimal even at float32 resolution
        ref, _ = reference_lloyd_step(X, C, labels)
        ref_err = max(ref_err, float(np.max(np.abs(ours.astype(np.float64) - ref)) / scale))
        empties += n_empty
        if vs_sklearn and n_empty:
            sk = KMeans(n_clusters=len(C), init=C, n_init=1, max_iter=1, tol=0.0, algorithm="lloyd").fit(X)
            err = float(np.max(np.abs(ours.astype(np.float64) - sk.cluster_centers_)) / scale)
            if err > 1e-5:  # allow a float32-level tie label: then inertia must still agree
                a, b = km.assign_mlx(parts, ours)[2], km.assign_mlx(parts, sk.cluster_centers_.astype(np.float32))[2]
                err = 0.0 if abs(a - b) / b <= 1e-6 else err
            sk_err = max(sk_err, err)
        C = ours
    passed = empties > 0 and ref_err <= 1e-6 and sk_err <= 1e-5
    extra = f"  vs scikit-learn {sk_err:.1e}" if vs_sklearn else ""
    print(f"{'PASS' if passed else 'FAIL'}  {name:50s} {empties} empty clusters relocated in {steps} steps  "
          f"center err vs float64 reference {ref_err:.1e}{extra}")
    return passed


def check_bounds(name, X, C0, steps=10, slice_rows=None, inject_at=None, rng=None):
    """Consecutive Lloyd steps through the per-centre bounds path (dims >= 256): the same checks as check_relocation
    at every step, plus the path must actually engage. With inject_at, unrelated centres replace the sequence at
    that step - the n_init restart - and the pass must fall back to a full evaluation and stay exact.

    The labels are read from the path's own state rather than from a second assign_mlx call: a step that rebuilt
    the bounds ran the tiles kernel, the next assignment runs the bounds kernel, and the two sum a distance in a
    different order, so a float32-level tie can land differently. Both are exact; only the step's own labels
    reproduce its centres.
    """
    mx = km._mx()
    per = slice_rows or len(X)
    parts = [mx.array(X[a:a + per]) for a in range(0, len(X), per)]
    scale = float(np.sqrt((X.astype(np.float64) ** 2).mean()))
    core._bounds_state.clear()
    C, ref_err, on_bounds, empties = C0.copy(), 0.0, 0, 0
    for step in range(steps):
        if inject_at is not None and step == inject_at:
            C = X[rng.choice(len(X), len(C), replace=False)].copy()
        for st in core._bounds_state.values():
            st.pop("visited", None)
        ours, _, n_empty = km.lloyd_step(parts, C)
        on_bounds += any("visited" in st for st in core._bounds_state.values())
        st = next(iter(core._bounds_state.values()))
        labels = np.concatenate([np.array(c) for chunks in st["labels"] for c in chunks]).astype(np.int64)
        _, ref_best = reference(X, C)
        X64, C64 = X.astype(np.float64), C.astype(np.float64)
        d_lab = ((X64 - C64[labels]) ** 2).sum(1)
        if (d_lab > ref_best + 8 * EPS32 * ((X64 ** 2).sum(1) + (C64[labels] ** 2).sum(1))).any():
            ref_err = float("inf")
        ref, _ = reference_lloyd_step(X, C, labels)
        ref_err = max(ref_err, float(np.max(np.abs(ours.astype(np.float64) - ref)) / scale))
        empties += n_empty
        C = ours
    passed = ref_err <= 1e-6 and on_bounds >= steps // 3          # engaged, not a policy pin: full passes precede it
    print(f"{'PASS' if passed else 'FAIL'}  {name:50s} {on_bounds}/{steps} steps on bounds, {empties} relocated  "
          f"center err vs float64 reference {ref_err:.1e}")
    return passed


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
    X = rng.random((60_000, 128), dtype=np.float32) * 255  # SIFT-like range, isotropic: nothing to prune
    results.append(check("uniform 0-255 d128 k1024", X, X[rng.choice(len(X), 1024, replace=False)]))
    X = blobs(rng, 120_000, 128, 64, 4, 1)                 # anisotropic: the cascade's prefix bites here
    X *= np.linspace(8.0, 0.05, 128).astype(np.float32)    # variance concentrated in the leading dims
    results.append(check("anisotropic d128 k256", X, X[rng.choice(len(X), 256, replace=False)]))
    X = blobs(rng, 20_000, 960, 50, 1, 1)                  # GIST-like dims
    results.append(check("high dims d960 k304", X, X[rng.choice(len(X), 304, replace=False)]))
    X = blobs(rng, 60_000, 64, 64, 1, 1)                   # overlapping at 64 dims: near-ties for the tiles path
    results.append(check("overlapping d64 k256", X, X[rng.choice(len(X), 256, replace=False)]))
    X = np.concatenate([blobs(rng, 30_000, 64, 8, 10, 1), np.zeros((16, 64), np.float32)])  # zero rows + tail rows
    results.append(check("zero rows + tail d64 k128", X[:30_009], X[rng.choice(30_000, 128, replace=False)]))
    X = blobs(rng, 250_003, 16, 16, 5, 1)
    results.append(check("many slices + chunks d16 k100", X, X[rng.choice(len(X), 100, replace=False)],
                         slice_rows=60_001, dist_bytes=4 * 100 * 7_777))
    # accumulation stress: ~5M points per cluster at |x| ~ 1000, one slice, few large blocks
    X = (blobs(rng, 20_000_000, 2, 4, 10, 1) + 1000).astype(np.float32)
    results.append(check("accumulation 20M rows +1000 d2 k4", X, X[rng.choice(len(X), 4, replace=False)]))
    # empty clusters: several empty at once (pairing order matters) -> checked against the float64 reference
    X = blobs(rng, 200_000, 16, 8, 10, 1)
    C0 = np.concatenate([X[rng.choice(len(X), 24, replace=False)], rng.uniform(900, 1000, (8, 16)).astype(np.float32)])
    results.append(check_relocation("relocation, 8 empty at once d16 k32", X, C0))
    X = blobs(rng, 300_000, 50, 64, 10, 1)
    C0 = np.concatenate([X[rng.choice(len(X), 60, replace=False)], rng.uniform(900, 1000, (4, 50)).astype(np.float32)])
    results.append(check_relocation("relocation, 4 empty at once d50 k64", X, C0))
    # one cluster empties -> also checked against scikit-learn itself
    X = blobs(rng, 200_000, 12, 16, 10, 1)
    C0 = np.concatenate([X[rng.choice(len(X), 31, replace=False)], rng.uniform(900, 1000, (1, 12)).astype(np.float32)])
    results.append(check_relocation("relocation, 1 empty d12 k32 (+ scikit-learn)", X, C0, vs_sklearn=True))
    # per-centre bounds between iterations (dims >= 256): exact at every step, including a restart mid-sequence
    X = blobs(rng, 60_000, 960, 50, 1, 1)
    results.append(check_bounds("bounds d960 k1024, restart at step 6", X, X[rng.choice(len(X), 1024, replace=False)],
                                steps=12, inject_at=6, rng=rng))
    X = blobs(rng, 50_000, 256, 40, 3, 1)
    results.append(check_bounds("bounds, 3 slices d256 k64", X, X[rng.choice(len(X), 64, replace=False)],
                                steps=8, slice_rows=20_001))
    X = blobs(rng, 80_000, 256, 30, 10, 1)
    C0 = np.concatenate([X[rng.choice(len(X), 60, replace=False)], rng.uniform(900, 1000, (4, 256)).astype(np.float32)])
    results.append(check_bounds("bounds with relocation d256 k64", X, C0, steps=8))
    print(f"\n{sum(results)}/{len(results)} cases passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
