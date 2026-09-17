#!/usr/bin/env python3
"""Decision-relevant comparison: quality per second, and ANN recall, not time per iteration.

bench.py compares one Lloyd pass at equal work. That is the right way to compare implementations, but it is not how
people use these libraries: FAISS trains an IVF quantizer on a subsample (max_points_per_centroid=256, so 256k of 1M
rows at k=1024) and stops after 25 iterations. This script answers two practical questions on SIFT1M:

  1. Quality per second: for each way of producing k centroids, wall-clock training time vs the inertia those
     centroids achieve on the FULL dataset (always measured by our float64-verified exact pass, so the metric is
     identical for everyone).
  2. ANN recall: build a FAISS IVF index from each centroid set and measure recall@10 against the ground-truth
     neighbours shipped with SIFT1M, plus search time, at several nprobe values.

Run:  .venv/bin/python bench_ann.py [--k 1024] [--queries 10000]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import kmeans as km
from bench import machine, other_cpu_percent, read_fvecs, git_state

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "benchmarks" / "ann_results.jsonl"
NPROBE = [1, 8, 32]
FAISS_NITER = 25          # FAISS's default niter
FAISS_SAMPLE = 256        # FAISS's default max_points_per_centroid


def read_ivecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    return raw.reshape(-1, d + 1)[:, 1:].copy()


def inertia_full(parts, C):
    """Inertia of these centroids over the whole dataset, by our exact pass (same metric for every method)."""
    return km.assign_mlx(parts, np.ascontiguousarray(C, dtype=np.float32))[2]


# ---------------------------------------------------------------- centroid producers

def ours(parts, rows, k, seed, sample=None, max_iter=50, tol=1e-4):
    """Our k-means to convergence (optionally on a FAISS-style subsample)."""
    rng = np.random.default_rng(seed)
    train = parts
    if sample:
        idx = rng.choice(rows, size=min(sample, rows), replace=False)
        train = [km._mx().array(km.take_rows(parts, np.sort(idx)))]
    C = km.kmeans_pp_init(train, train[0].shape[0] if sample else rows, k, rng)
    prev = None
    for it in range(max_iter):
        C, inertia, _ = km.lloyd_step(train, C)
        if prev is not None and abs(prev - inertia) <= tol * prev:
            break
        prev = inertia
    return C, it + 1


def faiss_kmeans(X, k, seed, subsample, niter=FAISS_NITER):
    import faiss
    m = faiss.Kmeans(X.shape[1], k, niter=niter, seed=seed, verbose=False,
                     max_points_per_centroid=FAISS_SAMPLE if subsample else 2**31 - 1)
    m.train(X)
    return m.centroids.copy(), niter


def sklearn_kmeans(X, k, seed, max_iter=FAISS_NITER):
    from sklearn.cluster import KMeans
    m = KMeans(n_clusters=k, n_init=1, max_iter=max_iter, tol=1e-4, random_state=seed).fit(X)
    return m.cluster_centers_.astype(np.float32), int(m.n_iter_)


# ---------------------------------------------------------------- ANN evaluation

def ivf_recall(base, queries, gt, C, k):
    """IVF index built from these centroids -> {nprobe: (recall@10, search seconds)} plus index build time."""
    import faiss
    d = base.shape[1]
    quantizer = faiss.IndexFlatL2(d)
    quantizer.add(np.ascontiguousarray(C, dtype=np.float32))
    index = faiss.IndexIVFFlat(quantizer, d, k)
    index.is_trained = True                      # centroids come from the caller, not from FAISS's own training
    t = time.perf_counter()
    index.add(base)
    build = time.perf_counter() - t
    out = {}
    for nprobe in NPROBE:
        index.nprobe = nprobe
        t = time.perf_counter()
        _, ids = index.search(queries, 10)
        search = time.perf_counter() - t
        hits = sum(len(set(ids[i]) & set(gt[i, :10])) for i in range(len(queries)))
        out[nprobe] = (hits / (10 * len(queries)), search)
    return out, build


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=1024)
    p.add_argument("--queries", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    mx = km._mx()
    base = read_fvecs(ROOT / "data/sift/sift_base.fvecs")
    queries = read_fvecs(ROOT / "data/sift/sift_query.fvecs")[:a.queries]
    gt = read_ivecs(ROOT / "data/sift/sift_groundtruth.ivecs")[:a.queries]
    parts = [mx.array(base)]
    mx.eval(parts)
    rows, k = len(base), a.k
    sample_rows = FAISS_SAMPLE * k
    print(f"SIFT1M: {rows:,} x {base.shape[1]}, k={k}, {len(queries):,} queries; FAISS-style subsample = {sample_rows:,} rows\n")

    methods = {
        f"ours, all {rows//1000}k rows, to convergence": lambda: ours(parts, rows, k, a.seed),
        f"ours, {sample_rows//1000}k-row subsample (FAISS-style)": lambda: ours(parts, rows, k, a.seed, sample=sample_rows),
        "ours, all rows, 25 iters (FAISS's budget)": lambda: ours(parts, rows, k, a.seed, max_iter=FAISS_NITER, tol=0.0),
        "FAISS default (subsample, 25 iters)": lambda: faiss_kmeans(base, k, a.seed, subsample=True),
        "FAISS all rows, 25 iters": lambda: faiss_kmeans(base, k, a.seed, subsample=False),
        "scikit-learn all rows, 25 iters": lambda: sklearn_kmeans(base, k, a.seed),
    }
    commit, dirty = git_state()
    rows_out = []
    for name, fn in methods.items():
        load = other_cpu_percent()
        t = time.perf_counter()
        C, iters = fn()
        train_s = time.perf_counter() - t
        inertia = inertia_full(parts, C)
        recall, build_s = ivf_recall(base, queries, gt, C, k)
        rec = dict(date=time.strftime("%Y-%m-%dT%H:%M:%S"), commit=commit, dirty=dirty, machine=machine(),
                   method=name, k=k, rows=rows, iters=iters, train_s=train_s, inertia=inertia,
                   index_build_s=build_s, other_cpu=load,
                   recall={str(n): recall[n][0] for n in NPROBE}, search_s={str(n): recall[n][1] for n in NPROBE})
        rows_out.append(rec)
        print(f"{name:46s} train {train_s:7.2f}s ({iters:2d} iters)  inertia {inertia:.6e}  index build {build_s:5.2f}s")
        print("      recall@10 " + "  ".join(f"nprobe {n}: {recall[n][0]*100:5.2f}% ({recall[n][1]:.2f}s)" for n in NPROBE), flush=True)
    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("a") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    best = min(rows_out, key=lambda r: r["inertia"])
    print(f"\nlowest inertia: {best['method']} ({best['inertia']:.6e})")
    for r in rows_out:
        print(f"  {r['method']:46s} inertia {r['inertia'] / best['inertia']:.4f}x best   train {r['train_s']:7.2f}s   "
              f"recall@10 nprobe 8 {r['recall']['8'] * 100:5.2f}%")


if __name__ == "__main__":
    main()
