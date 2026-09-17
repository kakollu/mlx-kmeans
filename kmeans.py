#!/usr/bin/env python3
"""K-means (Lloyd's algorithm) on a randomly initialized array, built to scale to 1B rows.

Memory: float32 data = rows * dims * 4 bytes  (1B x 8 dims = 32 GB).

Backends:
  mlx   (default) one fused Metal GPU kernel per pass: distance + argmin + per-cluster sums in a
        single read of the data, no intermediate buffers. Data is also generated on the GPU.
  numpy CPU reference: chunked expanded-distance math, never materializes the (rows x k) matrix.

Usage:
  python3 kmeans.py --rows 1_000_000_000 --dims 8 --k 16
  python3 kmeans.py --rows 100_000_000 --backend numpy
"""
import argparse
import time

import numpy as np

SLICE_ROWS = 100_000_000  # data is held as slices of at most this many rows


def slice_rows(dims):
    """Rows per slice: at most SLICE_ROWS, and few enough that row * dims fits the kernels' uint32 indices."""
    return min(SLICE_ROWS, (2**32 - 1) // dims)


def block_rows(n, k, d, max_out_bytes=64 << 20):
    """Rows per GPU thread: 4096, doubled while the per-block results for n rows would exceed ~64 MB
    (at large k * dims, 4096-row blocks would need gigabytes of result buffer)."""
    bs = 4096
    while -(-n // bs) * k * (d + 2) * 4 > max_out_bytes:
        bs *= 2
    return bs


# ---------------------------------------------------------------- data

def make_data_numpy(rows, dims, k, seed):
    """Random array: points scattered around k hidden centers, generated in chunks to cap peak memory."""
    rng = np.random.default_rng(seed)
    true_centers = rng.uniform(-10, 10, size=(k, dims)).astype(np.float32)
    X = np.empty((rows, dims), dtype=np.float32)
    chunk = 10_000_000
    for s in range(0, rows, chunk):
        e = min(s + chunk, rows)
        labels = rng.integers(0, k, size=e - s)
        X[s:e] = true_centers[labels] + rng.standard_normal((e - s, dims), dtype=np.float32)
    return [X]


_GEN_HEADER = """
static inline ulong mix64(ulong z) {  // splitmix64 finalizer: counter-based RNG, no shared state
    z += 0x9E3779B97F4A7C15ul;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    return z ^ (z >> 31);
}
static inline float u01(ulong h) { return (float(h >> 40) + 0.5f) / 16777216.0f; }
"""
# Row r = true_center[random label] + standard normal noise (Box-Muller), all in one GPU pass.
# About 3x faster than composing mx.random calls, which materialize labels, a gather and the noise separately.
_GEN_SRC = """
    uint i = thread_position_in_grid.x;
    ulong base = mix64(seed[0] ^ mix64(row0[0] + i));
    uint lab = uint(mix64(base) % K);
    for (uint j = 0; j < D; j++) {
        ulong h = mix64(base + 2 * j + 1);
        float u1 = u01(h), u2 = u01(mix64(h));
        X[i * D + j] = T[lab * D + j] + sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI_F * u2);
    }
"""
_gen_kernel = None


def make_data_mlx(rows, dims, k, seed):
    """Same distribution, generated on the GPU as a list of slices (no 2x-memory concatenate)."""
    global _gen_kernel
    mx = _mx()
    if _gen_kernel is None:
        _gen_kernel = mx.fast.metal_kernel(name="kmeans_gen", input_names=["T", "row0", "seed"],
                                           output_names=["X"], source=_GEN_SRC, header=_GEN_HEADER)
    rng = np.random.default_rng(seed)
    T = mx.array(rng.uniform(-10, 10, size=(k, dims)).astype(np.float32))
    parts = []
    per = slice_rows(dims)
    for s in range(0, rows, per):
        n = min(per, rows - s)
        part = _gen_kernel(inputs=[T, mx.array([s], dtype=mx.uint64), mx.array([seed], dtype=mx.uint64)],
                           template=[("D", dims), ("K", k)], grid=(n, 1, 1), threadgroup=(64, 1, 1),
                           output_shapes=[(n, dims)], output_dtypes=[mx.float32])[0]
        mx.eval(part)
        parts.append(part)
    return parts


def take_rows(parts, idx):
    """Gather global row indices from a list of slices -> float32 numpy array."""
    idx = np.asarray(idx)
    out = np.empty((len(idx), parts[0].shape[1]), dtype=np.float32)
    offset = 0
    for p in parts:
        m = (idx >= offset) & (idx < offset + p.shape[0])
        if m.any():
            local = idx[m] - offset
            out[m] = p[local] if isinstance(p, np.ndarray) else np.array(p[_mx().array(local)])
        offset += p.shape[0]
    return out


def _mx():
    import mlx.core as mx
    return mx


# ---------------------------------------------------------------- init

def kmeans_pp_init(parts, rows, k, rng, sample=200_000):
    """k-means++ seeding on a random subsample (full k-means++ over 1B rows is needlessly slow)."""
    S = take_rows(parts, rng.choice(rows, size=min(sample, rows), replace=False))
    C = np.empty((k, S.shape[1]), dtype=np.float32)
    C[0] = S[rng.integers(len(S))]
    d2 = ((S - C[0]) ** 2).sum(1)
    for i in range(1, k):
        C[i] = S[rng.choice(len(S), p=d2 / d2.sum())]
        d2 = np.minimum(d2, ((S - C[i]) ** 2).sum(1))
    return C


# ---------------------------------------------------------------- assignment passes

def assign_numpy(parts, C, chunk=2_000_000):
    """One Lloyd pass: nearest-center labels -> per-cluster sums/counts/inertia."""
    k, d = C.shape
    sums = np.zeros((k, d), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    inertia = 0.0
    c_sq = (C * C).sum(1)
    for X in parts:
        for s in range(0, len(X), chunk):
            x = X[s:s + chunk]
            # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2 ; ||x||^2 is constant per row so skip it for argmin
            dist = c_sq[None, :] - 2.0 * (x @ C.T)
            lab = dist.argmin(1)
            x_sq = np.einsum("ij,ij->i", x, x)
            inertia += float((x_sq + dist[np.arange(len(x)), lab]).sum())
            counts += np.bincount(lab, minlength=k)
            for j in range(d):
                sums[:, j] += np.bincount(lab, weights=x[:, j], minlength=k)
    return sums, counts, inertia


# Each GPU thread owns one block of BS rows and writes its own (K, D+2) row of [count, inertia, sums...],
# so there is no shared-write contention; the small per-block results are reduced on the host in float64.
_METAL_SRC = """
    uint b = thread_position_in_grid.x;
    uint start = b * BS;
    uint end = min(start + BS, n_rows[0]);
    uint W = D + 2;
    uint base = b * K * W;
    for (uint c = 0; c < K * W; c++) out[base + c] = 0;
    for (uint i = start; i < end; i++) {
        float best = INFINITY; uint bl = 0;
        for (uint c = 0; c < K; c++) {
            float s = 0;
            for (uint j = 0; j < D; j++) { float t = X[i*D + j] - C[c*D + j]; s += t*t; }
            if (s < best) { best = s; bl = c; }
        }
        uint o = base + bl * W;
        out[o] += 1; out[o + 1] += best;
        for (uint j = 0; j < D; j++) out[o + 2 + j] += X[i*D + j];
    }
"""
_kernel = None


def assign_mlx(parts, C):
    global _kernel
    mx = _mx()
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(name="kmeans_step", input_names=["X", "C", "n_rows"],
                                       output_names=["out"], source=_METAL_SRC)
    k, d = C.shape
    Cm = mx.array(C)
    tot = np.zeros((k, d + 2), dtype=np.float64)
    for x in parts:
        n = x.shape[0]
        block = block_rows(n, k, d)
        nb = -(-n // block)
        out = _kernel(inputs=[x, Cm, mx.array([n], dtype=mx.uint32)],
                      template=[("D", d), ("K", k), ("BS", block)],
                      grid=(nb, 1, 1), threadgroup=(64, 1, 1),
                      output_shapes=[(nb, k, d + 2)], output_dtypes=[mx.float32])[0]
        tot += np.array(out, dtype=np.float64).sum(0)
    return tot[:, 2:], tot[:, 0].astype(np.int64), float(tot[:, 1].sum())


# ---------------------------------------------------------------- driver

def kmeans(parts, rows, k, max_iter, tol, seed, backend):
    rng = np.random.default_rng(seed)
    C = kmeans_pp_init(parts, rows, k, rng)
    assign = assign_mlx if backend == "mlx" else assign_numpy
    prev = None
    for it in range(1, max_iter + 1):
        t = time.perf_counter()
        sums, counts, inertia = assign(parts, C)
        empty = counts == 0
        newC = C.copy()
        newC[~empty] = (sums[~empty] / counts[~empty, None]).astype(np.float32)
        if empty.any():  # re-seed empty clusters from random points
            newC[empty] = take_rows(parts, rng.integers(0, rows, size=int(empty.sum())))
        shift = float(np.sqrt(((newC - C) ** 2).sum(1)).max())
        C = newC
        print(f"iter {it:3d}  inertia {inertia:.6e}  max_center_shift {shift:.5f}  {time.perf_counter() - t:.3f}s")
        if prev is not None and abs(prev - inertia) <= tol * prev:
            break
        prev = inertia
    return C, inertia


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
