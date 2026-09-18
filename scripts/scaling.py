#!/usr/bin/env python3
"""Measure time per k-means pass as the data grows: ours vs scikit-learn vs FAISS, identical data and centers.

    .venv/bin/python scripts/scaling.py        # writes benchmarks/scaling.json for the README figure
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx_kmeans as km  # noqa: E402

ROWS = [1_000_000, 10_000_000, 100_000_000]
D, K, ITERS = 32, 64, 5


def timed(fn, reps=3):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main():
    out = []
    for rows in ROWS:
        parts = km.make_data_mlx(rows, D, K, 0)
        X = np.concatenate([np.array(p) for p in parts])
        C = km.kmeans_pp_init(parts, rows, K, np.random.default_rng(0))
        ours = timed(lambda: km.assign_mlx(parts, C))

        from sklearn.cluster import KMeans as SK
        sk = timed(lambda: SK(n_clusters=K, init=C, n_init=1, max_iter=ITERS, tol=0.0, algorithm="lloyd").fit(X),
                   reps=1) / (ITERS + 1)          # scikit-learn runs one extra assignment pass

        import faiss
        def run_faiss():
            m = faiss.Kmeans(D, K, niter=ITERS, max_points_per_centroid=2**31 - 1, verbose=False)
            m.train(X, init_centroids=C.copy())
        fa = timed(run_faiss, reps=1) / ITERS

        out.append(dict(rows=rows, ours=ours, sklearn=sk, faiss=fa))
        print(f"{rows:>12,}: ours {ours*1000:8.2f} ms   scikit-learn {sk*1000:9.2f} ms ({sk/ours:5.1f}x)   "
              f"faiss {fa*1000:9.2f} ms ({fa/ours:5.1f}x)", flush=True)
        del parts, X
        km._mx().clear_cache()

    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    (Path(__file__).resolve().parents[1] / "benchmarks/scaling.json").write_text(
        json.dumps(dict(dims=D, k=K, iters=ITERS, points=out, machine=chip), indent=1))
    print("wrote benchmarks/scaling.json")


if __name__ == "__main__":
    main()
