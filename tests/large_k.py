#!/usr/bin/env python3
"""Large k on a large slice: the cascade's pruning mask (one bit per row and centre) must be split into row chunks.

Before 2026-10-06, k = 65,536 on a 1.2M-row slice sized one launch at 1.2M x 2,048 words: past MLX's 32-bit shape limit,
which aborts the process (found by mlx-vsearch training an IVF index). Checked here: the cascade completes, every label
it gives matches an exact GPU path or is a float32-level tie in float64, and counts and inertia agree.
Run:  python3 tests/large_k.py        (about a minute on an M5 Max; exit code 1 on failure)
"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx_kmeans as km  # noqa: E402

EPS32 = float(np.finfo(np.float32).eps)
rng = np.random.default_rng(1)
n, d, k = 1_200_000, 128, 65_536
centres = rng.standard_normal((4096, d)).astype(np.float32) * 4
X = (centres[rng.integers(0, 4096, n)] + rng.standard_normal((n, d)).astype(np.float32)) * np.linspace(4, 0.25, d).astype(np.float32)
C = X[rng.choice(n, k, replace=False)].copy()
mx = km._mx()
parts = [mx.array(X[s:s + km.slice_rows(d)]) for s in range(0, n, km.slice_rows(d))]
t = time.perf_counter()
_, cnt_t, in_t, lab_t = km.assign_mlx(parts, C, method="tiles", return_labels=True)
t_tiles = time.perf_counter() - t
km.core._cascade_state.clear()
t = time.perf_counter()
_, cnt_c, in_c, lab_c = km.assign_mlx(parts, C, method="cascade", return_labels=True)
t_casc = time.perf_counter() - t
diff = np.flatnonzero(lab_t != lab_c)
X64, C64 = X[diff].astype(np.float64), C.astype(np.float64)
d_c = ((X64 - C64[lab_c[diff]]) ** 2).sum(1); d_t = ((X64 - C64[lab_t[diff]]) ** 2).sum(1)
tol = 8 * EPS32 * ((X64 ** 2).sum(1) + (C64[lab_c[diff]] ** 2).sum(1))
worse = int((d_c > d_t + tol).sum())
ok = worse == 0 and abs(in_c - in_t) <= 1e-6 * in_t and int(np.abs(cnt_c - cnt_t).sum()) <= 2 * len(diff)
print(f"{'PASS' if ok else 'FAIL'}  k {k:,} on {n:,} rows x {d}: cascade {t_casc:.1f} s (tiles {t_tiles:.1f} s); "
      f"{len(diff)} labels differ from tiles, {worse} of them not float32 ties; inertia rel diff {abs(in_c - in_t) / in_t:.1e}")
sys.exit(0 if ok else 1)
