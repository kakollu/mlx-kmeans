#!/usr/bin/env python3
"""K-means, kept simple for learning. Runs on the Apple GPU with MLX.

The fast version is kmeans.py (a custom GPU kernel, ~15x faster per pass); this one uses plain MLX ops.

Usage:  python3 kmeans_simple.py [rows]        (default 100M rows; 1B rows needs ~32 GB)
"""
import sys
import time

import mlx.core as mx
import numpy as np

ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000_000
DIMS = 8               # numbers per point
K = 16                 # number of clusters
CHUNK = 10_000_000     # rows handled at once on the GPU; keeps temporary memory small
MAX_ITER = 50
TOL = 1e-5             # stop when the error improves by less than this fraction

# ---- 1. Make random data: points scattered around K hidden centers --------------------------------
t0 = time.perf_counter()
mx.random.seed(0)
true_centers = mx.random.uniform(-10, 10, (K, DIMS))
chunks = []
for start in range(0, ROWS, CHUNK):
    n = min(CHUNK, ROWS - start)
    labels = mx.random.randint(0, K, (n,))                          # which hidden center each point belongs to
    chunk = true_centers[labels] + mx.random.normal((n, DIMS))      # center + noise
    mx.eval(chunk)                                                   # MLX is lazy: compute it now
    chunks.append(chunk)
print(f"{ROWS:,} rows generated in {time.perf_counter() - t0:.1f}s")

# ---- 2. Pick starting centers with k-means++ on a sample --------------------------------------------
# Purely random starting points often land two centers in one cluster and get stuck there.
# k-means++ picks each next center with probability proportional to its squared distance from the chosen ones.
rng = np.random.default_rng(0)
sample = np.array(chunks[0][:100_000])
init = [sample[rng.integers(len(sample))]]
for _ in range(K - 1):
    d2 = ((sample[:, None, :] - np.array(init)[None]) ** 2).sum(axis=2).min(axis=1)  # distance to nearest chosen center
    init.append(sample[rng.choice(len(sample), p=d2 / d2.sum())])
centers = mx.array(np.array(init, dtype=np.float32))

# ---- 3. Lloyd's algorithm: assign each point to its nearest center, move centers to the mean ------
t0 = time.perf_counter()
prev_error = None
for it in range(1, MAX_ITER + 1):
    t = time.perf_counter()
    # Running totals live in float64 on the CPU: float32 loses precision adding up 100M+ numbers.
    sums = np.zeros((K, DIMS))
    counts = np.zeros(K)
    error = 0.0
    for x in chunks:
        # Squared distance to every center, via |x - c|^2 = |x|^2 - 2 x.c + |c|^2.
        # |x|^2 is the same for all centers, so it can be dropped when we only want the closest one.
        dist = (centers * centers).sum(axis=1) - 2 * (x @ centers.T)       # shape (n, K)
        nearest = mx.argmin(dist, axis=1)                                   # shape (n,)

        chunk_sums = mx.zeros((K, DIMS)).at[nearest].add(x)                 # add each point to its cluster
        chunk_counts = mx.zeros((K,)).at[nearest].add(1)
        chunk_error = ((x - centers[nearest]) ** 2).sum()                  # exact squared distance
        mx.eval(chunk_sums, chunk_counts, chunk_error)

        sums += np.array(chunk_sums)
        counts += np.array(chunk_counts)
        error += chunk_error.item()

    # New center = mean of its points (an empty cluster keeps its old center).
    new_centers = np.array(centers)
    has_points = counts > 0
    new_centers[has_points] = sums[has_points] / counts[has_points, None]
    centers = mx.array(new_centers.astype(np.float32))

    print(f"iter {it:2d}  error {error:.6e}  {time.perf_counter() - t:.2f}s")
    if prev_error is not None and prev_error - error <= TOL * prev_error:
        break
    prev_error = error

print(f"done in {time.perf_counter() - t0:.1f}s")
print("first 3 centers:\n", np.round(np.array(centers[:3]), 2))
