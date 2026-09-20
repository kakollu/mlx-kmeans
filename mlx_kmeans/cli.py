"""Cluster a local file, or explicitly request a synthetic benchmark with --rows."""
import argparse
import time

import numpy as np

from .core import kmeans, make_data_mlx, make_data_numpy


def main():
    p = argparse.ArgumentParser(description='Cluster a local CSV/TSV/XLSX file. Start with a filename to inspect its columns.')
    p.add_argument('file', nargs='?', help='local CSV, TSV, or XLSX file')
    p.add_argument('--columns', nargs='+', help='numeric feature columns; IDs and text stay in the output only')
    p.add_argument('--output', help='new output folder (never overwrites an existing folder)')
    p.add_argument('--sheet', help='XLSX sheet name; default first sheet')
    p.add_argument('--n-init', '--nstart', type=int, default=10, help='number of starts (default 10)')
    p.add_argument('--missing', choices=['error','drop'], default='error')
    p.add_argument('--no-standardize', action='store_true', help='use original units instead of standardized features')
    p.add_argument("--rows", type=int, help='generate synthetic rows instead of reading a file')
    p.add_argument("--dims", type=int, default=8)
    p.add_argument("--k", type=int, help='number of clusters (files: 8; synthetic: 16)')
    p.add_argument("--max-iter", '--iter-max', type=int, help='iteration limit (files: 300; synthetic: 50)')
    p.add_argument("--tol", type=float, help="relative inertia change to stop (files: 1e-4; synthetic: 1e-5)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["mlx", "numpy"], default="mlx")
    a = p.parse_args()

    if a.file:
        if a.rows is not None:
            p.error('Choose a local file or --rows, not both.')
        if a.backend != 'mlx':
            p.error('--backend applies only to synthetic benchmarks.')
        from .files import cluster_file, inspect_file
        try:
            if not a.columns:
                preview = inspect_file(a.file,a.sheet)
                print('Columns:', ', '.join(preview.columns))
                print(preview.to_string(index=False))
                print('\nChoose numeric features with --columns and a group count with --k. '
                      'Do not include IDs or category codes as numeric features.')
                return
            result = cluster_file(a.file,columns=a.columns,k=8 if a.k is None else a.k,output=a.output,n_init=a.n_init,
                                  max_iter=300 if a.max_iter is None else a.max_iter,
                                  tol=1e-4 if a.tol is None else a.tol,seed=a.seed,
                                  standardize=not a.no_standardize,missing=a.missing,sheet=a.sheet)
        except (ValueError, OSError, ImportError) as e:
            p.error(str(e))
        run = result['run']
        print(f"Clustered {run['clustered_rows']:,} rows; {run['excluded_rows']:,} excluded; "
              f"fit including labels {run['fit_including_labels_s']:.3f}s.")
        print(result['summary'].to_string(index=False))
        print(f"Saved labeled.csv, cluster_summary.csv, run.json, model.npz to {result['output']}")
        if run['constant_features']:
            print('Constant features (no separating power):', ', '.join(run['constant_features']))
        return
    if a.rows is None:
        p.print_help()
        return
    a.max_iter = 50 if a.max_iter is None else a.max_iter
    a.k = 16 if a.k is None else a.k
    a.tol = 1e-5 if a.tol is None else a.tol

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
