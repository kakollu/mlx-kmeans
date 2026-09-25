#!/usr/bin/env python3
"""Compare exact assignment against the opt-in approximate path (KMeans(exact=False)).

    python3 scripts/compare_exact_approx.py            # speed + clustering quality
    python3 scripts/compare_exact_approx.py --ann      # also recall@10 through a FAISS IVF index

Three questions, kept separate because they answer differently:

  1. Speed. One assignment pass, both paths, same centres.
  2. How often the approximation picks a different centre. Measured at IDENTICAL centres, so it isolates the
     approximation itself - comparing labels after two full fits conflates it with the two runs reaching
     different local minima, which happens for reasons that have nothing to do with precision.
  3. Whether the resulting clustering is worse. Inertia of both fits, scored by the same exact routine, and
     for a vector-search quantizer the thing that actually matters: recall@10 of an IVF index built on the
     centroids.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mlx_kmeans import KMeans, core            # noqa: E402
from bench import read_fvecs                   # noqa: E402

SHAPES = [("synthetic 500k x 384", 500_000, 384, 1024),
          ("synthetic 500k x 768", 500_000, 768, 1024),
          ("synthetic 200k x 960", 200_000, 960, 1024),
          ("synthetic 500k x 128", 500_000, 128, 1024)]


def med(f, r=3):
    f()
    ts = []
    for _ in range(r):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def one_pass(name, X, k):
    """Speed and disagreement for a single pass from identical centres."""
    n, d = X.shape
    parts = [mx.array(X)]
    C = core.kmeans_pp_init(parts, n, k, np.random.default_rng(0))
    Cm = mx.array(C)
    exact_m = ("tiles1" if core._tiles1_ok(d, k) else
               "tiles" if (k >= core.TILES_MIN_K and d >= core.TILES_MIN_DIMS and d % 8 == 0 and k % 8 == 0)
               else "pairs" if d >= core.PAIRS_MIN_DIMS else "rows")
    le, be = core._nearest(parts[0], Cm, exact_m)
    la, ba = core._nearest(parts[0], Cm, "approx")
    mx.eval(le, be, la, ba)
    te = med(lambda: mx.eval(core._nearest(parts[0], Cm, exact_m)))
    ta = med(lambda: mx.eval(core._nearest(parts[0], Cm, "approx")))
    differ = int(mx.sum(le != la))
    # how much worse the approximate choice is, where it differs, as a fraction of the distance
    worse = float(mx.sum(ba - be)) / float(mx.sum(be))
    used = "approx" if d >= core.APPROX_MIN_DIMS else "exact (flag ignored below %d dims)" % core.APPROX_MIN_DIMS
    print(f"{name:>22}{te*1e3:>9.1f}ms{ta*1e3:>9.1f}ms{te/ta:>7.2f}x{differ:>9,} /{n//1000:>4}k"
          f"{worse:>11.2e}   {used}")
    del parts


def fits(name, X, k, iters):
    """Full fits both ways, each scored by the exact routine."""
    out = {}
    for exact in (True, False):
        t = time.perf_counter()
        m = KMeans(n_clusters=k, random_state=0, max_iter=iters, exact=exact).fit(X)
        secs = time.perf_counter() - t
        parts = [mx.array(X)]
        inertia = core.assign_mlx(parts, m.cluster_centers_)[2]      # same exact scorer for both
        out[exact] = (secs, inertia, m.cluster_centers_)
        del parts
    (se, ie, _), (sa, ia, _) = out[True], out[False]
    d = X.shape[1]
    note = "" if d >= core.APPROX_MIN_DIMS else f"   flag ignored below {core.APPROX_MIN_DIMS} dims"
    print(f"{name:>22}{se:>9.2f}s{sa:>9.2f}s{se/sa:>7.2f}x   inertia {(ia - ie) / ie:+.3e} relative"
          f"  ({'approx worse' if ia > ie else 'approx better'}){note}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", action="store_true", help="also measure recall@10 on GIST1M through FAISS IVF")
    ap.add_argument("--iters", type=int, default=15)
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    print("One assignment pass, identical centres\n")
    print(f'{"shape":>22}{"exact":>11}{"approx":>11}{"speedup":>7}{"labels differ":>16}'
          f'{"extra inertia":>12}   path used')
    for name, n, d, k in SHAPES:
        X = (rng.standard_normal((n, d)) * 3).astype(np.float32); X[:n // 2] += 5
        one_pass(name, X, k)
        del X

    print(f"\nFull fits, {a.iters} iterations, both scored by the exact routine\n")
    print(f'{"shape":>22}{"exact":>10}{"approx":>10}{"speedup":>7}')
    for name, n, d, k in SHAPES[:3]:
        X = (rng.standard_normal((n, d)) * 3).astype(np.float32); X[:n // 2] += 5
        fits(name, X, k, a.iters)
        del X

    if a.ann:
        from bench_ann import ivf_recall
        k = 1024
        base = read_fvecs(ROOT / "data/gist/gist_base.fvecs")
        queries = read_fvecs(ROOT / "data/gist/gist_query.fvecs")
        gt = np.fromfile(ROOT / "data/gist/gist_groundtruth.ivecs", dtype=np.int32).reshape(-1, 101)[:, 1:]
        print(f"\nGIST1M {len(base):,} x {base.shape[1]}, k={k}: does the approximation cost recall?\n")
        res = fits(f"GIST1M k={k}", base, k, a.iters)
        for exact in (True, False):
            rec, build = ivf_recall(base, queries, gt, res[exact][2], k)
            label = "exact " if exact else "approx"
            print(f"   {label}  index build {build:5.1f}s   " +
                  "   ".join(f"nprobe {p}: {rec[p][0]*100:5.2f}%" for p in rec))


if __name__ == "__main__":
    sys.exit(main())
