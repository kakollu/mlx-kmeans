#!/usr/bin/env python3
"""K-means (Lloyd's algorithm) on a randomly initialized array, built to scale to 1B rows.

Memory: float32 data = rows * dims * 4 bytes  (1B x 8 dims = 32 GB).

Backends:
  mlx   (default) custom Metal GPU kernels: exact nearest-center search and compensated accumulation
        (see assign_mlx). Data is also generated on the GPU.
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


# One assignment pass on the GPU in two stages, all distances computed directly as sum((x - c)^2) in float32
# (never via |x|^2 - 2x.c + |c|^2, which cancels badly; MLX's GPU matmul is also ~5000x eps inexact).
#
# 1. Nearest center per row (lowest index on ties), two interchangeable paths:
#      rows:  one GPU thread per row loops over all centers with a branch-free argmin. Fastest in most shapes.
#      pairs: one GPU thread per (row, center) pair writes a chunk of distances, then one thread per row takes
#             the branch-free argmin. Faster at high dims with modest k.
#    GPUs run many threads in lockstep; the original design (one thread per 4096-row block, branchy argmin)
#    was 9-40x slower at large k * dims.
# 2. Accumulation: each GPU thread owns a block of rows and adds [count, inertia, sums...] per cluster with
#    compensated (Neumaier) summation, then one thread per output value reduces the blocks the same way.
#    Exact to float32 rounding regardless of block size; the host adds sum + compensation in float64.
_KAHAN = """
#define KAHAN_ADD(s, c, v) { float v_ = (v); float t_ = (s) + v_; \\
    (c) += select((v_ - t_) + (s), ((s) - t_) + v_, fabs(s) >= fabs(v_)); (s) = t_; }
"""
_ROWS_SRC = """
    uint i = thread_position_in_grid.x, xi = (row0[0] + i) * D;
    float best = INFINITY; uint bl = 0;
    for (uint c = 0; c < K; c++) {
        float s = 0;
        for (uint j = 0; j < D; j++) { float t = X[xi + j] - C[c * D + j]; s += t * t; }
        bool lt = s < best; best = select(best, s, lt); bl = select(bl, c, lt);
    }
    labels[i] = bl; best_d[i] = best;
"""
_PAIRS_SRC = """
    uint i = thread_position_in_grid.y, c = thread_position_in_grid.x;
    uint xi = (row0[0] + i) * D, ci = c * D;
    float s = 0;
    for (uint j = 0; j < D; j++) { float t = X[xi + j] - C[ci + j]; s += t * t; }
    dist[i * K + c] = s;
"""
_ARGMIN_SRC = """
    uint i = thread_position_in_grid.x, di = i * K;
    float best = INFINITY; uint bl = 0;
    for (uint c = 0; c < K; c++) { float s = dist[di + c]; bool lt = s < best; best = select(best, s, lt); bl = select(bl, c, lt); }
    labels[i] = bl; best_d[i] = best;
"""
_ACCUMULATE_SRC = """
    uint b = thread_position_in_grid.x, W = D + 2;
    uint start = b * BS, end = min(start + BS, n_rows[0]), base = b * K * W;
    for (uint e = 0; e < K * W; e++) { sums[base + e] = 0; comp[base + e] = 0; }
    for (uint i = start; i < end; i++) {
        uint o = base + labels[i] * W, xi = i * D;
        sums[o] += 1;                                   // exact: at most 2^24 rows per block
        KAHAN_ADD(sums[o + 1], comp[o + 1], best_d[i]);
        for (uint j = 0; j < D; j++) KAHAN_ADD(sums[o + 2 + j], comp[o + 2 + j], X[xi + j]);
    }
"""
_REDUCE_SRC = """
    uint e = thread_position_in_grid.x;
    float s = 0, c = 0;
    for (uint b = 0; b < n_blocks[0]; b++) { KAHAN_ADD(s, c, sums[b * KW + e]); c += comp[b * KW + e]; }
    total[e] = s; total_comp[e] = c;
"""
_kernels = {}
PAIRS_MIN_DIMS = 64                     # use the pairs path when dims >= 64 (measured crossover, benchmarks/NOTES.md)
DIST_BYTES = 512 << 20                  # largest (rows x k) float32 distance chunk in the pairs path
ACC_BYTES = 256 << 20                   # budget for per-block accumulation buffers
ACC_MAX_BLOCKS = 1024                   # GPU threads for accumulation


def _kernel(name, inputs, outputs, src, header=""):
    if name not in _kernels:
        _kernels[name] = _mx().fast.metal_kernel(name=name, input_names=inputs, output_names=outputs,
                                                 source=src, header=header)
    return _kernels[name]


def _u32(v):
    return _mx().array([v], dtype=_mx().uint32)


def _nearest(x, Cm, method):
    """Nearest center and its distance for every row of one slice -> (labels uint32, best float32)."""
    mx = _mx()
    n, d = x.shape
    k = Cm.shape[0]
    if method == "rows":
        kern = _kernel("kmeans_rows", ["X", "C", "row0"], ["labels", "best_d"], _ROWS_SRC)
        return kern(inputs=[x, Cm, _u32(0)], template=[("D", d), ("K", k)], grid=(n, 1, 1), threadgroup=(64, 1, 1),
                    output_shapes=[(n,), (n,)], output_dtypes=[mx.uint32, mx.float32])
    pairs = _kernel("kmeans_pairs", ["X", "C", "row0"], ["dist"], _PAIRS_SRC)
    argmin = _kernel("kmeans_argmin", ["dist"], ["labels", "best_d"], _ARGMIN_SRC)
    chunk = max(1, DIST_BYTES // (4 * k))
    labels, best = [], []
    for r0 in range(0, n, chunk):
        m = min(chunk, n - r0)
        dist = pairs(inputs=[x, Cm, _u32(r0)], template=[("D", d), ("K", k)], grid=(k, m, 1), threadgroup=(16, 16, 1),
                     output_shapes=[(m, k)], output_dtypes=[mx.float32])[0]
        lab, bst = argmin(inputs=[dist], template=[("K", k)], grid=(m, 1, 1), threadgroup=(64, 1, 1),
                          output_shapes=[(m,), (m,)], output_dtypes=[mx.uint32, mx.float32])
        mx.eval(lab, bst)  # bound memory: one distance chunk alive at a time
        labels.append(lab)
        best.append(bst)
    return (labels[0], best[0]) if len(labels) == 1 else (mx.concatenate(labels), mx.concatenate(best))


def _accumulate(x, labels, best, k):
    """Per-cluster [count, inertia, sums...] for one slice -> float64 numpy array (k, d + 2)."""
    mx = _mx()
    n, d = x.shape
    w = d + 2
    nb = max(-(-n // 2**24), min(ACC_MAX_BLOCKS, max(1, ACC_BYTES // (8 * k * w)), n))
    bs = -(-n // nb)
    nb = -(-n // bs)
    acc = _kernel("kmeans_accumulate", ["X", "labels", "best_d", "n_rows"], ["sums", "comp"], _ACCUMULATE_SRC, _KAHAN)
    sums, comp = acc(inputs=[x, labels, best, _u32(n)], template=[("D", d), ("K", k), ("BS", bs)],
                     grid=(nb, 1, 1), threadgroup=(min(nb, 64), 1, 1),
                     output_shapes=[(nb * k * w,), (nb * k * w,)], output_dtypes=[mx.float32, mx.float32])
    red = _kernel("kmeans_reduce", ["sums", "comp", "n_blocks"], ["total", "total_comp"], _REDUCE_SRC, _KAHAN)
    total, total_comp = red(inputs=[sums, comp, _u32(nb)], template=[("KW", k * w)],
                            grid=(k * w, 1, 1), threadgroup=(64, 1, 1),
                            output_shapes=[(k * w,), (k * w,)], output_dtypes=[mx.float32, mx.float32])
    return (np.array(total, dtype=np.float64) + np.array(total_comp, dtype=np.float64)).reshape(k, w)


def assign_mlx(parts, C, method="auto", return_labels=False):
    """One Lloyd pass on the GPU -> (per-cluster sums, counts, inertia[, labels])."""
    mx = _mx()
    k, d = C.shape
    if method == "auto":
        method = "pairs" if d >= PAIRS_MIN_DIMS else "rows"
    Cm = mx.array(C)
    tot = np.zeros((k, d + 2), dtype=np.float64)
    all_labels = []
    for x in parts:
        labels, best = _nearest(x, Cm, method)
        tot += _accumulate(x, labels, best, k)
        if return_labels:
            all_labels.append(np.array(labels).astype(np.int64))
    result = (tot[:, 2:], np.rint(tot[:, 0]).astype(np.int64), float(tot[:, 1].sum()))
    return result + (np.concatenate(all_labels),) if return_labels else result


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
