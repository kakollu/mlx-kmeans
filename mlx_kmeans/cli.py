"""Command line: python3 -m mlx_kmeans --rows 1_000_000_000 --dims 8 --k 16"""
import argparse
import time

import numpy as np

from .core import kmeans, make_data_mlx, make_data_numpy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=1000_000_000)
    p.add_argument("--dims", type=int, default=8)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--tol", type=float, default=1e-5, help="relative inertia change to stop")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["mlx", "numpy"], default="mlx")
    a = p.parse_args()

    print(f"rows={a.rows:,} dims={a.dims} k={a.k} backend={a.backend} "
          f"data≈{a.rows * a.dims * 4 / 1e9:.1f} GB")
    t = time.perf_counter()
    make = make_data_mlx if a.backend == "mlx" else make_data_numpy
    parts = make(a.rows, a.dims, a.k, a.seed)
    print(f"generated data in {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    C, inertia = kmeans(parts, a.rows, a.k, a.max_iter, a.tol, a.seed, a.backend)
    print(f"done in {time.perf_counter() - t:.1f}s  final inertia {inertia:.6e}")
    print("centers (first 5):\n", np.round(C[:5], 3))

if __name__ == "__main__":
    main()
