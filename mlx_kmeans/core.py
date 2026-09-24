"""K-means (Lloyd's algorithm) on Apple Silicon GPUs, built to scale to 1B rows.

Memory: float32 data = rows * dims * 4 bytes  (1B x 8 dims = 32 GB).

Backends:
  mlx   (default) custom Metal GPU kernels: exact nearest-center search and compensated accumulation
        (see assign_mlx). Data is also generated on the GPU.
  numpy CPU reference: chunked expanded-distance math, never materializes the (rows x k) matrix.

Usage:
  python3 kmeans.py --rows 1_000_000_000 --dims 8 --k 16
  python3 kmeans.py --rows 100_000_000 --backend numpy
"""
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
# Greedy k-means++ (what scikit-learn does): each round samples 2 + log(k) candidate rows with probability
# proportional to their squared distance d2, then keeps the candidate that lowers the total d2 the most. Picking a
# single candidate per round - the textbook version - converged to 2x worse inertia on clustered synthetic data.
# Two kernels per round, both one thread per (row, candidate), and no CPU round-trip: the chosen indices stay on
# the GPU until the centers are gathered at the end.
_PP_KEYS_SRC = """
    uint i = thread_position_in_grid.y, t = thread_position_in_grid.x;
    ulong h = mix64(seed[0] ^ mix64(i * TRIALS + t));
    // Gumbel-max: argmax over rows of log(d2) + Gumbel noise samples in proportion to d2
    key[i * TRIALS + t] = log(max(d2[i], 1e-30f)) - log(-log(u01(h)));
"""
_PP_EVAL_SRC = """
    uint i = thread_position_in_grid.y, t = thread_position_in_grid.x;
    uint xi = i * D, ci = cand[t] * D;
    float s = 0;
    for (uint j = 0; j < D; j++) { float q = S[xi + j] - S[ci + j]; s += q * q; }
    mins[i * TRIALS + t] = min(d2[i], s);
"""


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

def kmeans_pp_init(parts, rows, k, rng, sample=None):
    """k-means++ seeding on a random subsample (full k-means++ over 1B rows is needlessly slow).

    The sample defaults to 20k rows or 20 per center, whichever is larger (capped at 200k). Each of the k rounds
    re-reads the whole sample, so the cost is k * sample * dims bytes of memory traffic: at k=1024 on GIST1M, a
    200k-row sample costs 2.2 s and a 20k-row one 0.4 s, with final inertia differing by <0.03% on SIFT1M and GIST1M
    (on synthetic blobs the run-to-run spread from local minima, 10-30%, dwarfs any difference).

    The k sequential rounds run entirely on the GPU, and nothing crosses back to the host until the last one:
    each round's work is small, so forcing a sync per round made seeding latency-bound rather than work-bound
    (one round trip per center, measured at 0.25-0.42 ms against a 0.14 ms bare-round-trip floor: 16 ms of a
    19 ms fit at k=64 on 1k rows). Leaving the rounds lazy lets MLX pipeline them for the same centers bit for
    bit, 2-4x faster, at a bounded ~90 MB of extra working memory.
    Doing the rounds in NumPy instead costs ~10 s at k=1024 (and is why scikit-learn's own k-means++ takes
    minutes on SIFT1M); it only wins below a few thousand sample rows, which is not worth a second code path.
    """
    mx = _mx()
    if sample is None:
        sample = min(200_000, max(20_000, 20 * k))
    S = mx.array(take_rows(parts, np.sort(rng.choice(rows, size=min(sample, rows), replace=False))))
    m, d = S.shape
    trials = 2 + int(np.log(k)) if k > 1 else 1
    keys_k = _kernel("kmeans_pp_keys", ["d2", "seed"], ["key"], _PP_KEYS_SRC, _GEN_HEADER)
    eval_k = _kernel("kmeans_pp_eval", ["S", "d2", "cand"], ["mins"], _PP_EVAL_SRC)
    first = int(rng.integers(m))
    picks = [mx.array([first], dtype=mx.uint32)]
    d2 = ((S - S[first]) ** 2).sum(1)
    for _ in range(k - 1):
        key = keys_k(inputs=[d2, mx.array([int(rng.integers(2**63 - 1))], dtype=mx.uint64)],
                     template=[("TRIALS", trials)], grid=(trials, m, 1), threadgroup=(trials, 64 // trials or 1, 1),
                     output_shapes=[(m * trials,)], output_dtypes=[mx.float32])[0]
        cand = mx.argmax(key.reshape(m, trials), axis=0).astype(mx.uint32)
        mins = eval_k(inputs=[S, d2, cand], template=[("D", d), ("TRIALS", trials)],
                      grid=(trials, m, 1), threadgroup=(trials, 64 // trials or 1, 1),
                      output_shapes=[(m * trials,)], output_dtypes=[mx.float32])[0].reshape(m, trials)
        best = mx.argmin(mins.sum(0))                      # candidate that lowers total d2 the most
        d2 = mins[:, best]
        picks.append(cand[best][None])
    return np.array(S[mx.concatenate(picks)], dtype=np.float32)


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


# One assignment pass on the GPU in two stages. Every label is the center with the smallest float32 distance
# sum((x - c)^2), lowest index on ties, whichever path runs.
#
# 1. Nearest center per row, three interchangeable paths:
#      rows:  one GPU thread per row loops over all centers with a branch-free argmin, distances computed directly.
#             Fastest below ~64 dims.
#      pairs: one GPU thread per (row, center) pair writes a chunk of direct distances, then one thread per row takes
#             the branch-free argmin. For high dims when the tiles path does not apply.
#      tiles: Metal simdgroup matrices (8x8 hardware tiles) compute x.c for a chunk of rows -- IEEE-quality, measured
#             1.9x eps at 960 dims, same as Accelerate -- then one thread per row forms approximate distances
#             |c|^2 - 2x.c, keeps every center within the provable float32 error bound of the best one, and
#             recomputes those candidates exactly. ~3x faster than pairs at high dims and still exact. Needs dims and
#             k to be multiples of 8 (tail rows fall back to the rows kernel).
#    GPUs run many threads in lockstep; the original design (one thread per 4096-row block, branchy argmin)
#    was 9-43x slower at large k * dims.
#    Apple's Neural Accelerator matmul (what MLX's @ operator and PyTorch MPS use) is another ~4x faster than tiles,
#    but has no float32 error bound: exact on integers, up to ~12000x eps on fractional data. It is only used by the
#    opt-in fast path (see _NEAREST_FAST), never by the default exact paths.
# 2. Accumulation of [count, inertia, sums...] per cluster, compensated (Neumaier) summation throughout, so it is
#    exact to float32 rounding; the host adds sum + compensation in float64. Two interchangeable methods:
#      blocks: each GPU thread owns a block of rows and writes into its own (k, dims+2) buffer; then one thread
#              per output value reduces the blocks. Best when k * (dims+2) is small.
#      sorted: rows are argsorted by label and split into 256-row segments of one cluster; one GPU thread per
#              segment writes one small row, then one thread per (cluster, value) reduces its segments.
#              At large k * dims the blocks method's scattered writes into multi-MB buffers dominate the whole
#              pass (GIST1M: 4.2 s of 4.8 s); sorted segments keep writes cache-local (0.05 s).
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
# One simdgroup (32 threads) per 8x8 tile of x.c; Ct is C transposed (dims x k) so both loads are row-major.
_DOT_TILES_SRC = """
    uint sg = thread_position_in_grid.x / 32;
    uint tiles_c = K / 8;
    uint rt = sg / tiles_c, ct = sg % tiles_c;
    uint xrow = (row0[0] + rt * 8) * D, c0 = ct * 8;
    simdgroup_float8x8 a, b, acc = simdgroup_float8x8(0.0f);
    for (uint kk = 0; kk < D; kk += 8) {
        simdgroup_load(a, X + xrow + kk, D);
        simdgroup_load(b, Ct + kk * K + c0, K);
        simdgroup_multiply_accumulate(acc, a, b, acc);
    }
    simdgroup_store(acc, dot + rt * 8 * K + c0, K);
"""
# Same arithmetic with 2x2 register blocking: one simdgroup owns a 16x16 output block, so each X tile and each
# center tile it loads feeds two accumulators instead of one. That halves the tile loads per multiply-accumulate,
# which is what the 8x8 version is short of - measured 9.5 -> 12.7 TFLOP/s at 128 dims and 10.0 -> 14.5 at 960,
# against 14.4 and 36.8 for MLX's `@` (which is not usable here: ~6600x eps against this kernel's ~6x).
# Each output element still accumulates over D in the same order, so results are bit-identical to the 8x8 path.
_DOT_TILES16_SRC = """
    uint sg = thread_position_in_grid.x / 32;
    uint tiles_c = K / 16;
    uint rt = sg / tiles_c, ct = sg % tiles_c;
    uint xrow = (row0[0] + rt * 16) * D, c0 = ct * 16;
    simdgroup_float8x8 a0, a1, b0, b1;
    simdgroup_float8x8 acc00 = simdgroup_float8x8(0.0f), acc01 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc10 = simdgroup_float8x8(0.0f), acc11 = simdgroup_float8x8(0.0f);
    for (uint kk = 0; kk < D; kk += 8) {
        simdgroup_load(a0, X  + xrow + kk, D);
        simdgroup_load(a1, X  + xrow + 8 * D + kk, D);
        simdgroup_load(b0, Ct + kk * K + c0, K);
        simdgroup_load(b1, Ct + kk * K + c0 + 8, K);
        simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
        simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
        simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
        simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
    }
    simdgroup_store(acc00, dot + (rt * 16)     * K + c0,     K);
    simdgroup_store(acc01, dot + (rt * 16)     * K + c0 + 8, K);
    simdgroup_store(acc10, dot + (rt * 16 + 8) * K + c0,     K);
    simdgroup_store(acc11, dot + (rt * 16 + 8) * K + c0 + 8, K);
"""
# In float32, |c|^2 - 2 x.c is off by at most E_c = 3 * gamma_D * (|x|^2 + |c|^2), where gamma_D = D*eps/(1-D*eps)
# bounds the relative error of a D-term dot product and of |c|^2 (Higham), and sum|x_j c_j| <= (|x|^2+|c|^2)/2 leaves
# slack for the subtraction. So the truly nearest center c* satisfies approx_c* <= amin + E_cmin + E_c*: only centers
# inside that window can win, and each gets an exact distance.
_MARGIN_SRC = """
    uint i = thread_position_in_grid.x, xi = (row0[0] + i) * D, di = i * K;
    float xsq = 0;
    for (uint j = 0; j < D; j++) xsq += X[xi + j] * X[xi + j];
    float amin = INFINITY; uint cmin = 0;
    for (uint c = 0; c < K; c++) {
        float a = csq[c] - 2 * dot[di + c];
        bool lt = a < amin; amin = select(amin, a, lt); cmin = select(cmin, c, lt);
    }
    float g3 = 3 * gamma[0];
    float bound = amin + g3 * (xsq + csq[cmin]);
    float best = INFINITY; uint bl = 0;
    for (uint c = 0; c < K; c++) {
        if (csq[c] - 2 * dot[di + c] > bound + g3 * (xsq + csq[c])) continue;
        float s = 0;
        for (uint j = 0; j < D; j++) { float t = X[xi + j] - C[c * D + j]; s += t * t; }
        bool lt = s < best; best = select(best, s, lt); bl = select(bl, c, lt);
    }
    labels[i] = bl; best_d[i] = best;
"""
# Outputs are zero-filled by MLX (init_value=0); zeroing k*(dims+2) slots per thread inside the kernel cost more
# than the accumulation itself on small inputs (0.98 ms of a 1.6 ms pass at 100k x 32, k=64).
_ACCUMULATE_SRC = """
    uint b = thread_position_in_grid.x, W = D + 2;
    uint start = b * BS, end = min(start + BS, n_rows[0]), base = b * K * W;
    for (uint i = start; i < end; i++) {
        uint o = base + labels[i] * W, xi = i * D;
        sums[o] += 1;                                   // exact: at most 2^24 rows per block
        KAHAN_ADD(sums[o + 1], comp[o + 1], best_d[i]);
        for (uint j = 0; j < D; j++) KAHAN_ADD(sums[o + 2 + j], comp[o + 2 + j], X[xi + j]);
    }
"""
# Threadgroup accumulation (opt-in, accumulate="atomic"): one threadgroup per chunk of rows sums into on-chip memory
# (atomic float adds) and
# writes a single (k, dims+2) block. The blocks path has to zero-fill and re-read one buffer per thread - ~18 MB for
# 3.4M additions at 100k x 32, k=64 - which dominates small inputs. Needs k*(dims+2) floats to fit threadgroup memory
# (32 KB), and sums a chunk in plain float32 rather than compensated, so it is used only where that is accurate
# enough (see ATOMIC_MAX_ROWS_PER_BLOCK).
_ATOMIC_SRC = """
    threadgroup atomic_float tile[KW];
    uint lid = thread_position_in_threadgroup.x, b = threadgroup_position_in_grid.x, W = D + 2;
    for (uint e = lid; e < KW; e += TG) atomic_store_explicit(&tile[e], 0.0f, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint start = b * BS, end = min(start + BS, n_rows[0]);
    for (uint i = start + lid; i < end; i += TG) {
        uint o = labels[i] * W, xi = i * D;
        atomic_fetch_add_explicit(&tile[o], 1.0f, memory_order_relaxed);
        atomic_fetch_add_explicit(&tile[o + 1], best_d[i], memory_order_relaxed);
        for (uint j = 0; j < D; j++) atomic_fetch_add_explicit(&tile[o + 2 + j], X[xi + j], memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lid; e < KW; e += TG) sums[b * KW + e] = atomic_load_explicit(&tile[e], memory_order_relaxed);
"""
_SEGMENT_SRC = """
    uint s = thread_position_in_grid.x, W = D + 2, o = s * W;
    for (uint p = seg_start[s]; p < seg_end[s]; p++) {
        uint r = perm[p], xi = r * D;
        sums[o] += 1;
        KAHAN_ADD(sums[o + 1], comp[o + 1], best_d[r]);
        for (uint j = 0; j < D; j++) KAHAN_ADD(sums[o + 2 + j], comp[o + 2 + j], X[xi + j]);
    }
"""
_SEGMENT_REDUCE_SRC = """
    uint c = thread_position_in_grid.y, e = thread_position_in_grid.x;
    float s = 0, cc = 0;
    for (uint q = seg_first[c]; q < seg_first[c + 1]; q++) { KAHAN_ADD(s, cc, sums[q * W + e]); cc += comp[q * W + e]; }
    total[c * W + e] = s; total_comp[c * W + e] = cc;
"""
_REDUCE_SRC = """
    uint e = thread_position_in_grid.x;
    float s = 0, c = 0;
    for (uint b = 0; b < n_blocks[0]; b++) { KAHAN_ADD(s, c, sums[b * KW + e]); c += comp[b * KW + e]; }
    total[e] = s; total_comp[e] = c;
"""
_kernels = {}
PAIRS_MIN_DIMS = 40                     # dims from which the pairs/tiles paths beat the rows path (measured)
TILES_MIN_K = 64                        # ...and the k from which tiles beats pairs (measured; below it pairs wins
                                        # by a lot at high dims - 960 dims, k=32: pairs 36.6 ms vs tiles 51.8 ms)
TILE_THREADGROUP = 512                  # threads per threadgroup in the tiles path (measured best)
DIST_BYTES = 512 << 20                  # largest (rows x k) float32 distance chunk in the pairs path
ACC_BYTES = 256 << 20                   # budget for per-block accumulation buffers
ACC_ROWS_PER_BLOCK = 256                # target rows per accumulation thread (measured sweet spot)
ACC_MIN_BLOCKS, ACC_MAX_BLOCKS = 1024, 4096   # more blocks = less work per thread but more to reduce
ATOMIC_MAX_KW = 8192                    # k*(dims+2) floats must fit threadgroup memory (32 KB)
ATOMIC_MAX_ROWS_PER_BLOCK = 4096        # rows summed in plain float32 per threadgroup; keeps the error ~1e-7
ATOMIC_THREADGROUP = 256
SORTED_MIN_KW = 1024                    # k*(dims+2) from which sorting by label can beat per-block buffers
SORTED_MAX_ROWS = 2_000_000             # above this the argsort costs more than the buffers it saves...
SORTED_ALWAYS_KW = 4096                 # ...unless the buffers are this large, where blocks is hopeless                    # use sorted accumulation when k * (dims + 2) >= this (measured crossover)
SEGMENT_ROWS = 256                      # rows per GPU thread in sorted accumulation


def _kernel(name, inputs, outputs, src, header=""):
    if name not in _kernels:
        _kernels[name] = _mx().fast.metal_kernel(name=name, input_names=inputs, output_names=outputs,
                                                 source=src, header=header)
    return _kernels[name]


def _u32(v):
    return _mx().array([v], dtype=_mx().uint32)


def _rows_kernel(x, Cm, r0, m, k, d):
    mx = _mx()
    kern = _kernel("kmeans_rows", ["X", "C", "row0"], ["labels", "best_d"], _ROWS_SRC)
    return kern(inputs=[x, Cm, _u32(r0)], template=[("D", d), ("K", k)], grid=(m, 1, 1), threadgroup=(64, 1, 1),
                output_shapes=[(m,), (m,)], output_dtypes=[mx.uint32, mx.float32])


def _tiles_nearest(x, Cm, n, k, d):
    mx = _mx()
    eps = float(np.finfo(np.float32).eps)
    csq = (Cm * Cm).sum(1)
    Ct = mx.array(np.ascontiguousarray(np.array(Cm).T))
    gamma = mx.array([d * eps / (1 - d * eps)], dtype=mx.float32)
    # The blocked kernel needs whole 16-row tiles and k a multiple of 16; otherwise fall back to 8x8 tiles.
    step = 16 if k % 16 == 0 else 8
    dot_k = (_kernel("kmeans_dot_tiles16", ["X", "Ct", "row0"], ["dot"], _DOT_TILES16_SRC) if step == 16
             else _kernel("kmeans_dot_tiles", ["X", "Ct", "row0"], ["dot"], _DOT_TILES_SRC))
    margin = _kernel("kmeans_margin", ["X", "C", "csq", "dot", "row0", "gamma"], ["labels", "best_d"], _MARGIN_SRC)
    chunk = max(step, (DIST_BYTES // (4 * k)) // step * step)
    whole = n // step * step                 # simdgroup tiles cover whole tiles; tail rows use the rows kernel
    labels, best = [], []
    for r0 in range(0, whole, chunk):
        m = min(chunk, whole - r0)
        dot = dot_k(inputs=[x, Ct, _u32(r0)], template=[("D", d), ("K", k)],
                    grid=((m // step) * (k // step) * 32, 1, 1), threadgroup=(TILE_THREADGROUP, 1, 1),
                    output_shapes=[(m * k,)], output_dtypes=[mx.float32])[0]
        lab, bst = margin(inputs=[x, Cm, csq, dot, _u32(r0), gamma], template=[("D", d), ("K", k)],
                          grid=(m, 1, 1), threadgroup=(64, 1, 1),
                          output_shapes=[(m,), (m,)], output_dtypes=[mx.uint32, mx.float32])
        mx.eval(lab, bst)                    # bound memory: one dot chunk alive at a time
        labels.append(lab)
        best.append(bst)
    if whole < n:
        lab, bst = _rows_kernel(x, Cm, whole, n - whole, k, d)
        labels.append(lab)
        best.append(bst)
    return (labels[0], best[0]) if len(labels) == 1 else (mx.concatenate(labels), mx.concatenate(best))


def _nearest(x, Cm, method):
    """Nearest center and its distance for every row of one slice -> (labels uint32, best float32)."""
    mx = _mx()
    n, d = x.shape
    k = Cm.shape[0]
    if method == "tiles":
        return _tiles_nearest(x, Cm, n, k, d)
    if method == "rows":
        return _rows_kernel(x, Cm, 0, n, k, d)
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


def _accumulate(x, labels, best, k, method="auto"):
    """Per-cluster [count, inertia, sums...] for one slice -> float64 numpy array (k, d + 2)."""
    w, n = x.shape[1] + 2, x.shape[0]
    if method == "atomic" and k * w > ATOMIC_MAX_KW:
        method = "auto"                       # the tile would not fit threadgroup memory
    if method == "auto":
        # "atomic" is faster but not bit-reproducible (threadgroup atomics fix no summation order), and in an
        # iterative algorithm a 1e-9 difference can flip a near-tie label and change the local minimum reached.
        # It stays opt-in; the default paths give identical results run to run.
        #
        # Between the two deterministic paths: sorting pays for itself when the per-block buffers are large
        # (k*(dims+2) >= 1024) but the argsort is O(n log n), so at many rows with a small k the blocks path wins -
        # measured at 10M rows: blocks 5.8 ms vs sorted 9.0 ms for k*w = 1536.
        big_buffers = k * w >= SORTED_MIN_KW
        method = "sorted" if big_buffers and (n <= SORTED_MAX_ROWS or k * w >= SORTED_ALWAYS_KW) else "blocks"
    if method == "atomic":
        return _accumulate_atomic(x, labels, best, k)
    return _accumulate_sorted(x, labels, best, k) if method == "sorted" else _accumulate_blocks(x, labels, best, k)


def _accumulate_atomic(x, labels, best, k):
    mx = _mx()
    n, d = x.shape
    w, kw = d + 2, k * (d + 2)
    bs = min(ATOMIC_MAX_ROWS_PER_BLOCK, max(256, -(-n // 2048)))
    nb = -(-n // bs)
    kern = _kernel("kmeans_atomic", ["X", "labels", "best_d", "n_rows"], ["sums"], _ATOMIC_SRC)
    sums = kern(inputs=[x, labels, best, _u32(n)],
                template=[("D", d), ("KW", kw), ("BS", bs), ("TG", ATOMIC_THREADGROUP)],
                grid=(nb * ATOMIC_THREADGROUP, 1, 1), threadgroup=(ATOMIC_THREADGROUP, 1, 1),
                output_shapes=[(nb * kw,)], output_dtypes=[mx.float32])[0]
    return np.array(sums, dtype=np.float64).reshape(nb, k, w).sum(0)


def _accumulate_sorted(x, labels, best, k):
    mx = _mx()
    n, d = x.shape
    w = d + 2
    perm = mx.argsort(labels).astype(mx.uint32)
    counts = np.array(mx.zeros((k,), dtype=mx.uint32).at[labels].add(mx.array(1, dtype=mx.uint32))).astype(np.int64)
    offs = np.concatenate([[0], np.cumsum(counts)])
    nseg = -(-counts // SEGMENT_ROWS)                      # segments per cluster (0 for an empty cluster)
    seg_first = np.concatenate([[0], np.cumsum(nseg)])
    seg_cluster = np.repeat(np.arange(k), nseg)
    starts = offs[seg_cluster] + (np.arange(len(seg_cluster)) - seg_first[seg_cluster]) * SEGMENT_ROWS
    ends = np.minimum(starts + SEGMENT_ROWS, offs[seg_cluster + 1])
    ns = len(starts)
    seg = _kernel("kmeans_segments", ["X", "best_d", "perm", "seg_start", "seg_end"], ["sums", "comp"], _SEGMENT_SRC, _KAHAN)
    sums, comp = seg(inputs=[x, best, perm, mx.array(starts.astype(np.uint32)), mx.array(ends.astype(np.uint32))],
                     template=[("D", d)], grid=(ns, 1, 1), threadgroup=(64, 1, 1), init_value=0,
                     output_shapes=[(ns * w,), (ns * w,)], output_dtypes=[mx.float32, mx.float32])
    red = _kernel("kmeans_segment_reduce", ["sums", "comp", "seg_first"], ["total", "total_comp"], _SEGMENT_REDUCE_SRC, _KAHAN)
    total, total_comp = red(inputs=[sums, comp, mx.array(seg_first.astype(np.uint32))], template=[("W", w)],
                            grid=(w, k, 1), threadgroup=(32, 8, 1),
                            output_shapes=[(k * w,), (k * w,)], output_dtypes=[mx.float32, mx.float32])
    return (np.array(total, dtype=np.float64) + np.array(total_comp, dtype=np.float64)).reshape(k, w)


def _accumulate_blocks(x, labels, best, k):
    mx = _mx()
    n, d = x.shape
    w = d + 2
    nb = min(max(-(-n // ACC_ROWS_PER_BLOCK), ACC_MIN_BLOCKS), ACC_MAX_BLOCKS)   # ~256 rows per thread
    nb = max(-(-n // 2**24), min(nb, max(1, ACC_BYTES // (8 * k * w)), n))       # memory cap; counts stay exact
    bs = -(-n // nb)
    nb = -(-n // bs)
    acc = _kernel("kmeans_accumulate", ["X", "labels", "best_d", "n_rows"], ["sums", "comp"], _ACCUMULATE_SRC, _KAHAN)
    sums, comp = acc(inputs=[x, labels, best, _u32(n)], template=[("D", d), ("K", k), ("BS", bs)],
                     grid=(nb, 1, 1), threadgroup=(min(nb, 64), 1, 1), init_value=0,
                     output_shapes=[(nb * k * w,), (nb * k * w,)], output_dtypes=[mx.float32, mx.float32])
    red = _kernel("kmeans_reduce", ["sums", "comp", "n_blocks"], ["total", "total_comp"], _REDUCE_SRC, _KAHAN)
    total, total_comp = red(inputs=[sums, comp, _u32(nb)], template=[("KW", k * w)],
                            grid=(k * w, 1, 1), threadgroup=(64, 1, 1),
                            output_shapes=[(k * w,), (k * w,)], output_dtypes=[mx.float32, mx.float32])
    return (np.array(total, dtype=np.float64) + np.array(total_comp, dtype=np.float64)).reshape(k, w)


def _gpu_pass(parts, C, method="auto", accumulate="auto"):
    """One assignment pass -> (float64 totals (k, d+2), [(labels, best) per slice] as MLX arrays)."""
    mx = _mx()
    k, d = C.shape
    if method == "auto":
        # Measured on 2M-10M rows: the rows kernel wins below ~40 dims at every k; above that the choice turns on
        # k, not on dims. The old rule used dims alone, so it sent 48-dim data to rows (1.6x slower than tiles at
        # k=128) and sent small-k high-dim data to tiles (1.6x slower than pairs at 64 dims, k=8).
        if d < PAIRS_MIN_DIMS:
            method = "rows"
        elif k >= TILES_MIN_K and d % 8 == 0 and k % 8 == 0:
            method = "tiles"
        else:
            method = "pairs"
    Cm = mx.array(C)
    tot = np.zeros((k, d + 2), dtype=np.float64)
    nearest = []
    for x in parts:
        labels, best = _nearest(x, Cm, method)
        tot += _accumulate(x, labels, best, k, accumulate)
        nearest.append((labels, best))
    return tot, nearest


def assign_mlx(parts, C, method="auto", return_labels=False, accumulate="auto"):
    """One Lloyd pass on the GPU -> (per-cluster sums, counts, inertia[, labels])."""
    tot, nearest = _gpu_pass(parts, C, method, accumulate)
    result = (tot[:, 2:], np.rint(tot[:, 0]).astype(np.int64), float(tot[:, 1].sum()))
    if return_labels:
        result += (np.concatenate([np.array(lab).astype(np.int64) for lab, _ in nearest]),)
    return result


def lloyd_step(parts, C, method="auto", accumulate="auto"):
    """One exact Lloyd iteration on the GPU -> (new centers float32, inertia, number of empty clusters).

    Empty clusters follow scikit-learn's rules (_relocate_empty_clusters_dense, _average_centers): each empty cluster
    takes one of the points farthest from their assigned centers, which leaves its old cluster; a cluster still empty
    after that (only when every point sits on its center) goes to the largest cluster's center. One deliberate
    difference: when several clusters empty at once, scikit-learn pairs them with far points in np.argpartition's
    unspecified order; here the farthest point goes to the lowest-numbered empty cluster (ties: lower row index).
    """
    tot, nearest = _gpu_pass(parts, C, method, accumulate)
    sums, counts, inertia = tot[:, 2:], np.rint(tot[:, 0]).astype(np.int64), float(tot[:, 1].sum())
    empty = np.flatnonzero(counts == 0)
    if len(empty):
        best = np.concatenate([np.array(b) for _, b in nearest])        # float32 distance to assigned center
        if best.max() > 0:
            far = np.argpartition(best, -len(empty))[-len(empty):]
            far = far[np.lexsort((far, -best[far]))]                    # farthest first, lower row index on ties
            labels = np.concatenate([np.array(lab) for lab, _ in nearest]).astype(np.int64)
            for new, f, row in zip(empty, far, take_rows(parts, far).astype(np.float64)):
                old = labels[f]
                sums[old] -= row
                counts[old] -= 1
                sums[new] = row
                counts[new] = 1
    del nearest
    C_new = np.empty_like(C)
    has = counts > 0
    C_new[has] = sums[has] / counts[has, None]
    C_new[~has] = C_new[np.argmax(counts)]
    return C_new, inertia, len(empty)


# ---------------------------------------------------------------- driver

def kmeans(parts, rows, k, max_iter, tol, seed, backend):
    rng = np.random.default_rng(seed)
    C = kmeans_pp_init(parts, rows, k, rng)
    prev = None
    for it in range(1, max_iter + 1):
        t = time.perf_counter()
        if backend == "mlx":
            newC, inertia, n_empty = lloyd_step(parts, C)
        else:  # NumPy CPU reference: empty clusters re-seeded from random points
            sums, counts, inertia = assign_numpy(parts, C)
            empty = counts == 0
            n_empty = int(empty.sum())
            newC = C.copy()
            newC[~empty] = (sums[~empty] / counts[~empty, None]).astype(np.float32)
            if n_empty:
                newC[empty] = take_rows(parts, rng.integers(0, rows, size=n_empty))
        shift = float(np.sqrt(((newC - C) ** 2).sum(1)).max())
        C = newC
        note = f"  ({n_empty} empty clusters relocated)" if n_empty else ""
        print(f"iter {it:3d}  inertia {inertia:.6e}  max_center_shift {shift:.5f}  {time.perf_counter() - t:.3f}s{note}")
        if prev is not None and abs(prev - inertia) <= tol * prev:
            break
        prev = inertia
    # Each pass reports the inertia of the centers it started from, so a run that stops before convergence would
    # otherwise return a stale number (measured 0.31% off at 4 capped iterations). Score the centers being returned.
    inertia = assign_mlx(parts, C)[2] if backend == "mlx" else assign_numpy(parts, C)[2]
    return C, inertia
