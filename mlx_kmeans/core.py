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
import weakref

import numpy as np

SLICE_ROWS = 100_000_000  # data is held as slices of at most this many rows
PART_MAX_ELEMENTS = 2**32 - 1  # and at most this many elements: the kernels address one slice with 32-bit offsets


def slice_rows(dims):
    """Rows per slice: at most SLICE_ROWS, and few enough that row * dims fits the kernels' uint32 indices."""
    return min(SLICE_ROWS, PART_MAX_ELEMENTS // dims)


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
# keys and their argmax in one dispatch: one threadgroup per trial reduces over the whole sample on-chip,
# instead of writing an (m x trials) key array for a separate mx.argmax to read back. Ties keep the first row,
# matching mx.argmax, so the candidates are identical.
_PP_PICK_SRC = """
    threadgroup float bv[256];
    threadgroup uint  bi[256];
    uint t = threadgroup_position_in_grid.x;
    uint lid = thread_position_in_threadgroup.x;
    float best = -INFINITY; uint bestr = 0;
    for (uint i = lid; i < M; i += 256) {
        ulong h = mix64(seed[0] ^ mix64(i * TRIALS + t));
        float key = log(max(d2[i], 1e-30f)) - log(-log(u01(h)));
        bool gt = key > best; best = select(best, key, gt); bestr = select(bestr, i, gt);
    }
    bv[lid] = best; bi[lid] = bestr;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint sft = 128; sft > 0; sft >>= 1) {
        if (lid < sft) {
            bool gt = bv[lid + sft] > bv[lid];
            bv[lid] = select(bv[lid], bv[lid + sft], gt);
            bi[lid] = select(bi[lid], bi[lid + sft], gt);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) cand[t] = bi[0];
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
    pick_k = _kernel("kmeans_pp_pick", ["d2", "seed"], ["cand"], _PP_PICK_SRC, _GEN_HEADER)
    eval_k = _kernel("kmeans_pp_eval", ["S", "d2", "cand"], ["mins"], _PP_EVAL_SRC)
    first = int(rng.integers(m))
    picks = [mx.array([first], dtype=mx.uint32)]
    d2 = ((S - S[first]) ** 2).sum(1)
    # All the per-round seeds in one transfer: drawing them one at a time cost a host allocation per round.
    seeds = mx.array(rng.integers(2**63 - 1, size=k - 1).astype(np.uint64))
    for r in range(k - 1):
        cand = pick_k(inputs=[d2, seeds[r:r + 1]], template=[("M", m), ("TRIALS", trials)],
                      grid=(trials * 256, 1, 1), threadgroup=(256, 1, 1),
                      output_shapes=[(trials,)], output_dtypes=[mx.uint32])[0]
        mins = eval_k(inputs=[S, d2, cand], template=[("D", d), ("TRIALS", trials)],
                      grid=(trials, m, 1), threadgroup=(trials, 64 // trials or 1, 1),
                      output_shapes=[(m * trials,)], output_dtypes=[mx.float32])[0].reshape(m, trials)
        best = mx.argmin(mins.sum(0))                      # candidate that lowers total d2 the most
        d2 = mins[:, best]
        picks.append(cand[best][None])
    return _host(S[mx.concatenate(picks)], np.float32)


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
# The loop above re-reads the row from cache once per center: k*D loads where D would do. Holding the row in
# registers first is the same arithmetic in the same order - bit-identical output - and measured 1.0-3.2x faster
# across d=4..96, k=8..1024, with the gain growing with k. Above ~96 dims the array spills and it turns into a
# 1.7x loss, so `_rows_kernel` only uses this below ROWS_REG_MAX_DIMS; the tail rows of the high-dimensional
# tiles/pairs paths keep the original.
_ROWS_REG_SRC = """
    uint i = thread_position_in_grid.x, xi = (row0[0] + i) * D;
    float xv[D];
    for (uint j = 0; j < D; j++) xv[j] = X[xi + j];
    float best = INFINITY; uint bl = 0;
    for (uint c = 0; c < K; c++) {
        float s = 0;
        for (uint j = 0; j < D; j++) { float t = xv[j] - C[c * D + j]; s += t * t; }
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
# which is what the 8x8 version is short of - measured 9.5 -> 12.7 TFLOP/s at 128 dims and 10.0 -> 14.5 at 960.
# There is no more to get here: a loop of simdgroup_multiply_accumulate with no loads at all, enough
# accumulators to fill the pipeline and enough simdgroups to fill the GPU, tops out at 15.7 TFLOP/s on this
# machine, so this kernel runs at 94% of what the instruction can do. MLX's `@` reaches 28.7 at 960 dims by
# using different hardware, and is not usable here: ~5900x eps against this kernel's ~9x.
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
# The obvious form scans the dot row twice: once for the minimum, once to apply the bound. The row is the
# largest thing this kernel reads, so the second scan is expensive. Instead, one scan carries the running minimum
# and records every center passing the RUNNING bound - never tighter than the final one, so the recorded list is a
# superset of the true candidates. The list is then filtered by the final bound, leaving the same ~1 center per row
# to recompute exactly. Output is bit-identical to the two-scan form; measured 1.34-1.56x faster.
# MARGIN_CANDIDATES only has to be large enough to make overflow rare (measured ~7 survive the running bound);
# a row that overflows falls back to a second scan, so correctness never depends on it.
_MARGIN_SRC = """
    uint i = thread_position_in_grid.x, xi = (row0[0] + i) * D, di = i * K;
    float xsq = 0;
    for (uint j = 0; j < D; j++) xsq += X[xi + j] * X[xi + j];
    float g3 = 3 * gamma[0];
    float amin = INFINITY; uint cmin = 0;
    uint cand[M]; uint nc = 0; bool overflow = false;
    for (uint c = 0; c < K; c++) {
        float a = csq[c] - 2 * dot[di + c];
        bool lt = a < amin; amin = select(amin, a, lt); cmin = select(cmin, c, lt);
        if (a <= amin + g3 * (xsq + csq[cmin]) + g3 * (xsq + csq[c])) {
            if (nc < M) { cand[nc] = c; nc++; } else { overflow = true; }
        }
    }
    float bound = amin + g3 * (xsq + csq[cmin]);
    float best = INFINITY; uint bl = 0;
    if (overflow) {
        for (uint c = 0; c < K; c++) {
            if (csq[c] - 2 * dot[di + c] > bound + g3 * (xsq + csq[c])) continue;
            float s = 0;
            for (uint j = 0; j < D; j++) { float t = X[xi + j] - C[c * D + j]; s += t * t; }
            bool lt = s < best; best = select(best, s, lt); bl = select(bl, c, lt);
        }
    } else {
        for (uint t = 0; t < nc; t++) {
            uint c = cand[t];
            if (csq[c] - 2 * dot[di + c] > bound + g3 * (xsq + csq[c])) continue;
            float s = 0;
            for (uint j = 0; j < D; j++) { float q = X[xi + j] - C[c * D + j]; s += q * q; }
            bool lt = s < best; best = select(best, s, lt); bl = select(bl, c, lt);
        }
    }
    labels[i] = bl; best_d[i] = best;
"""
# Outputs are zero-filled by MLX (init_value=0); zeroing k*(dims+2) slots per thread inside the kernel cost more
# than the accumulation itself on small inputs (0.98 ms of a 1.6 ms pass at 100k x 32, k=64).
# Threads stride through the rows rather than each taking a contiguous block. Same work and the same thread ->
# row mapping run to run, but adjacent threads now read adjacent rows instead of addresses BS*D*4 apart (58 KB at
# 10M rows), so the loads coalesce: measured 1.16-1.19x at low dims and 2.48x at 1M x 128, k=1024. The summation
# order per cluster changes, so totals differ from the contiguous version in the last bits (1e-12 relative or
# better, with Kahan compensation unchanged); the path stays deterministic.
_ACCUMULATE_SRC = """
    uint b = thread_position_in_grid.x, W = D + 2;
    uint base = b * K * W;
    for (uint i = b; i < n_rows[0]; i += NB) {
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
# One simdgroup per segment, with the running totals in registers. A segment belongs to exactly one cluster,
# so its accumulator is dims+2 floats - not k*(dims+2) - and ceil((dims+2)/32) of them fit in each lane's
# registers along with their Kahan carries. The row then costs one coalesced read of X and nothing else.
# The previous version gave a segment to a single thread and kept its totals in device memory, so every row
# read-modify-wrote dims+2 floats there, in both the sums and the carries: four times the traffic of the data
# itself, which is where the bandwidth went. Same arithmetic in the same order, so results are unchanged;
# measured 1.5x at 32 dims, 2.2x at 128, and 5.6x on GIST1M at 960 dims, where it lands within 10% of one
# pass over the data.
_SEGMENT_SRC = """
    uint s = thread_position_in_grid.x / 32, lid = thread_position_in_grid.x % 32;
    const uint W = D + 2;
    float acc[DPL], carry[DPL];
    for (uint c = 0; c < DPL; c++) { acc[c] = 0.0f; carry[c] = 0.0f; }
    for (uint p = seg_start[s]; p < seg_end[s]; p++) {
        uint r = perm[p], xi = r * D;
        for (uint c = 0; c < DPL; c++) {
            uint e = lid + c * 32;
            if (e < W) {
                float v = (e >= 2) ? X[xi + e - 2] : ((e == 0) ? 1.0f : best_d[r]);
                KAHAN_ADD(acc[c], carry[c], v);
            }
        }
    }
    for (uint c = 0; c < DPL; c++) {
        uint e = lid + c * 32;
        if (e < W) { sums[s * W + e] = acc[c]; comp[s * W + e] = carry[c]; }
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
PAIRS_MIN_DIMS = 80                     # dims from which the pairs path beats the rows path (measured)
TILES_MIN_DIMS = 80                     # ...and the dims from which tiles beats rows. Both moved up from
                                        # 64/40 once the rows kernel started holding the row in registers:
                                        # that made it 1.0-3.2x faster and invalidated a threshold measured
                                        # an hour earlier. rows now wins to 64 dims at every k tested, and
                                        # the old rule cost 1.46x at 48 dims k=256 and 1.26x at 64 dims.
FUSED_MAX_KW = 128                      # k*(dims+2) past which the fused kernel loses (measured)
FUSED_ROWS_PER_BLOCK = 4096             # rows per threadgroup; 512-65536 all measured within 25%
TG_FLOATS = 8192                        # threadgroup memory on Apple Silicon, in float32 slots
TILES1_KEEP = 4                         # candidate slots held in registers by the one-pass tiles path
TILES1_DIMS = (32, 96)                  # dims it is worth using over. Below 32 the reduction costs more than
                                        # the multiply it replaces, and zero-padding a short row wastes a
                                        # growing share of the tile (2x at 4 dims); above 96 the row tiles
                                        # spill out of registers and it loses 3x.
TILES1_RELOAD_DIMS = 224                # ...and how far the same path reaches once the row tiles are
                                        # re-read from cache instead of held in registers. Past this the
                                        # re-reads cost more than the n x k matrix they avoid (0.87x at 256).
TILES1_MIN_K = 64                       # ...and the k from which it wins consistently (measured; at k=32 it
                                        # ranges from 0.55x to 2.07x across dims, which is not worth picking)
TILES1_MAX_FALLBACK = 0.10              # simdgroup fallback rate past which this path stops being worth it. A
                                        # simdgroup that falls back scores all K centres exactly, which costs
                                        # about what the rows kernel costs for its 16 rows, so break-even is
                                        # near 1/3; this leaves a wide margin.
TILES1_PROBE_ROWS = 1 << 18             # rows a new dataset is measured on before the rest of the pass
CASCADE_MIN_DIMS = 128                  # below this the one-pass path already fuses the whole distance
CASCADE_MAX_DIMS = 256                  # above this the prefix cannot be a useful share of the row: the
                                        # prefix tiles have to fit in registers, so it caps at 96 dims,
                                        # which is 10% of GIST's 960 and prunes nothing (70% survive)
CASCADE_MIN_K = 256                     # and below this k there is too little to prune for the bound to pay
CASCADE_MAX_SURVIVORS = 0.15            # survivor fraction past which pruning costs more than it saves:
                                        # survivors are scored irregularly, at about a sixth of the dense
                                        # rate, so break-even is near 1/6 once the prefix pass is counted
_cascade_state = {}                     # per dataset: variance order, prefix buffer, last pass's labels
APPROX_MIN_DIMS = 256                   # dims from which giving up exactness buys anything at all. Below it
                                        # the exact paths win outright - the prefix-pruning cascade at 128
                                        # dims (0.76x) and 192 (0.95x) - and above it the approximate path
                                        # pulls ahead: 1.97x at 256, 1.96x at 384, 2.16x at 960 in fp32, and
                                        # more where the fp16 guard passes (GIST: 2.68x on a fit). This moved
                                        # down from 384 when the approximate path stopped walking the n x k
                                        # product four times; it is a property of both paths, so it has to
                                        # be re-measured whenever either changes.
TILES1_SIMDGROUPS = 4                   # simdgroups per threadgroup there (16 rows each)
MARGIN_CANDIDATES = 24                  # per-row candidate slots before falling back to a second scan
ROWS_REG_MAX_DIMS = 96                  # above this the per-thread row array spills registers (measured)
TILES_MIN_K = 64                        # ...and the k from which tiles beats pairs (measured; below it pairs wins
                                        # by a lot at high dims - 960 dims, k=32: pairs 36.6 ms vs tiles 51.8 ms)
TILE_THREADGROUP = 512                  # threads per threadgroup in the tiles path (measured best)
DIST_BYTES = 512 << 20                  # largest (rows x k) float32 distance chunk in the pairs path
BOUNDS_MIN_DIMS = 256                   # per-centre bounds: below this the bound traffic costs more than the multiply it saves
BOUNDS_MAX_DIMS = 2048                  # the step kernel holds a row in registers, D/32 per lane
BOUNDS_MAX_BYTES = 16 << 30             # n x k float32 lower bounds, plus one DIST_BYTES chunk while it is rewritten
BOUNDS_MAX_VISITED = 0.05               # centres the count pass expects to visit, as a fraction of n x k; above it, full pass
BOUNDS_SAMPLE_ROWS = 8192               # rows the count and prediction passes look at (strided); the step counts exactly
ACC_BYTES = 256 << 20                   # budget for per-block accumulation buffers
ACC_ROWS_PER_BLOCK = 256                # target rows per accumulation thread (measured sweet spot)
ACC_MIN_BLOCKS, ACC_MAX_BLOCKS = 1024, 4096   # more blocks = less work per thread but more to reduce
ATOMIC_MAX_KW = 8192                    # k*(dims+2) floats must fit threadgroup memory (32 KB)
ATOMIC_MAX_ROWS_PER_BLOCK = 4096        # rows summed in plain float32 per threadgroup; keeps the error ~1e-7
ATOMIC_THREADGROUP = 256
LANES_MAX_KW = TG_FLOATS // 2           # two k*(dims+2) tiles - the sums and their Kahan carries - on chip
LANES_BLOCKS = 512                      # simdgroups to split the rows over. What matters is the block COUNT,
                                        # not the rows per block: the best setting was ~500 blocks at 500k,
                                        # 2M and 10M rows alike, i.e. enough to fill the GPU and no more,
                                        # since every extra block re-zeroes and writes out a k*(dims+2) tile.
LANES_MIN_ROWS = 256                    # ...but not so few rows each that the tile overhead dominates
LANES_SMALL_KW = 1536                   # k*(dims+2) up to which lanes wins at every row count measured
LANES_ROWS_PER_SLOT = 10000             # ...and above it, the rows per slot from which it wins anyway.
                                        # Two costs trade off. The sorted path pays an argsort - 0.6 ns a
                                        # row, 6 ms at 10M, and slightly superlinear - but then accumulates
                                        # in registers. lanes needs no sort but re-zeroes and writes out a
                                        # k*(dims+2) tile per block, and a bigger tile also means fewer
                                        # threadgroups resident. So a small tile wins outright, and a large
                                        # one only once there are enough rows. Re-measured after the
                                        # counting sort made grouping 4-6x cheaper: sorted now beats lanes
                                        # at kw=2176 even at 10M rows (5.6 vs 7.3 ms), so this moved from
                                        # 4000 to 10000; lanes still wins outright below LANES_SMALL_KW.
LANES_ROWS_AHEAD = 8                    # rows loaded before any is added, to overlap their loads
SORTED_MIN_KW = 1024                    # k*(dims+2) from which sorting by label can beat per-block buffers
SORTED_MAX_ROWS = 2_000_000             # above this the argsort costs more than the buffers it saves...
SORTED_ALWAYS_KW = 4096                 # ...unless the buffers are this large, where blocks is hopeless                    # use sorted accumulation when k * (dims + 2) >= this (measured crossover)
SEGMENT_ROWS = 256                      # rows per GPU thread in sorted accumulation


def _held(parts):
    """-> weak references to each part: a cache entry's proof that it belongs to THIS data.

    id() is recycled the moment an array is freed, and the ordinary pattern - fit on one array, predict on the
    next one of the same shape - hands the second array the first one's id. An entry keyed on the id alone then
    serves the previous dataset's float16 copy, or its prefix and last labels, to the new one: wrong labels,
    silently. A weak reference cannot be recycled - it goes dead with the array - and a live one is checked
    by identity, part by part, so sliced inputs that share a first part are told apart too.
    """
    return [weakref.ref(p) for p in parts]


def _hold(cache, key, entry, parts):
    """Install entry under key, and drop it again the moment any of its arrays is freed.

    Without this the state - a float16 copy, the cascade prefix, n x k lower bounds - outlives the data it
    describes until the next dataset arrives: fit() builds its arrays from the caller's numpy and they die when
    it returns. The callback checks that the entry is still the one installed, so an older array dying late
    cannot evict a newer entry of the same shape.
    """
    def gone(_ref):
        if cache.get(key) is entry:
            cache.pop(key, None)
    entry["refs"] = [weakref.ref(p, gone) for p in parts]
    cache[key] = entry
    return entry


def _holds(st, parts):
    refs = st.get("refs")
    return refs is not None and len(refs) == len(parts) and all(r() is p for r, p in zip(refs, parts))


def _host(a, dtype=None):
    """MLX array -> numpy, with the evaluation forced BEFORE numpy asks for the buffer.

    Converting a lazy array lets numpy trigger the evaluation from inside its own C buffer request. If a
    Metal error is raised there - a kernel that will not load, a threadgroup allocation the device refuses -
    the C++ exception cannot propagate back through that frame and the process is terminated with SIGABRT
    instead of a Python exception. Eight such aborts came out of one audit run. Evaluating explicitly first
    turns every such failure into a catchable RuntimeError, which is also what makes the fallbacks in
    _gpu_pass able to fall back at all.
    """
    mx = _mx()
    mx.eval(a)
    return np.array(a, dtype=dtype) if dtype is not None else np.array(a)


def _kernel(name, inputs, outputs, src, header=""):
    if name not in _kernels:
        _kernels[name] = _mx().fast.metal_kernel(name=name, input_names=inputs, output_names=outputs,
                                                 source=src, header=header)
    return _kernels[name]


def _u32(v):
    return _mx().array([v], dtype=_mx().uint32)


def _rows_kernel(x, Cm, r0, m, k, d):
    mx = _mx()
    kern = (_kernel("kmeans_rows_reg", ["X", "C", "row0"], ["labels", "best_d"], _ROWS_REG_SRC)
            if d <= ROWS_REG_MAX_DIMS else
            _kernel("kmeans_rows", ["X", "C", "row0"], ["labels", "best_d"], _ROWS_SRC))
    return kern(inputs=[x, Cm, _u32(r0)], template=[("D", d), ("K", k)], grid=(m, 1, 1), threadgroup=(64, 1, 1),
                output_shapes=[(m,), (m,)], output_dtypes=[mx.uint32, mx.float32])


def _tiles_nearest(x, Cm, n, k, d, lb_out=None):
    mx = _mx()
    eps = float(np.finfo(np.float32).eps)
    csq = (Cm * Cm).sum(1)
    Ct = mx.array(np.ascontiguousarray(np.array(Cm).T))
    gamma_f = d * eps / (1 - d * eps)
    gamma = mx.array([gamma_f], dtype=mx.float32)
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
        lab, bst = margin(inputs=[x, Cm, csq, dot, _u32(r0), gamma],
                          template=[("D", d), ("K", k), ("M", MARGIN_CANDIDATES)],
                          grid=(m, 1, 1), threadgroup=(64, 1, 1),
                          output_shapes=[(m,), (m,)], output_dtypes=[mx.uint32, mx.float32])
        mx.eval(lab, bst)                    # bound memory: one dot chunk alive at a time
        if lb_out is not None:
            lb_out.append(_lb_from_dot(x, r0, dot, csq, m, k, d, gamma_f))
        labels.append(lab)
        best.append(bst)
    if whole < n:
        lab, bst = _rows_kernel(x, Cm, whole, n - whole, k, d)
        if lb_out is not None:
            lb_out.append(_lb_chunk(x, Cm, Ct, csq, whole, n, k, d, gamma_f))
        labels.append(lab)
        best.append(bst)
    return (labels[0], best[0]) if len(labels) == 1 else (mx.concatenate(labels), mx.concatenate(best))


# The tiles path above is fast at the multiply and slow everywhere else: it writes an n x k matrix of dot
# products to memory and reads it straight back, which at k=256 is most of the pass. This does the same
# multiply and never stores the matrix. Two things make that possible:
#
#   - the row tiles stay in simdgroup registers for the whole centre loop, so X is read once rather than
#     K/16 times (the reason a naive fusion is slower, not faster);
#   - the candidates are held in a fixed number of named registers instead of an indexed array. A candidate
#     array is the obvious way to write this and costs more than the multiply it is protecting: indexing it
#     by a running count forces it to scratch memory. Measured at 4M x 32, k=256: multiply plus running
#     minimum 5.3 ms, with an indexed array 11.3 ms, with registers 5.7 ms, against 10.9 ms for the scalar
#     kernel.
#
# Exactness is the margin kernel's argument with one change. Keeping only the R smallest approximate
# distances means the discarded centres are not individually tested, so the test uses the largest |c|^2 over
# all centres: every discarded centre has approx >= v[R-1], so if v[R-1] is above the bound widened by that
# maximum, none of them can be the true nearest and scoring the R kept centres settles the row. When it is
# not, the row is scored exactly against every centre. R = 4 because duplicate centres - which relocation and
# seeding on repeated rows can both produce - put several exact ties in the window at once: with half the
# centres duplicated R = 2 loses 3.2x, while R = 4 falls back on 0.8% of rows and stays level with the
# scalar kernel. Ties go to the lowest index, as in every other path, which needs an explicit comparison here
# because the kept set is ordered by distance rather than by index.
_SCORE_HDR = """
#define SCORE(cc) { uint c_ = (cc); float s = 0; \\
    for (uint j = 0; j < D; j++) { float q = X[xi + j] - C[c_ * D + j]; s += q * q; } \\
    bool lt = (s < best) || (s == best && c_ < bl); best = select(best, s, lt); bl = select(bl, c_, lt); }
"""


def _tiles1_src(r):
    """Source for the one-pass tiles kernel keeping the r nearest candidates in registers."""
    keep = "".join(f"                bool l{j} = a < v{j};\n" for j in range(r))
    for j in range(r - 1, 0, -1):
        keep += (f"                v{j} = l{j-1} ? v{j-1} : (l{j} ? a : v{j});\n"
                 f"                u{j} = l{j-1} ? u{j-1} : (l{j} ? c : u{j});\n")
    keep += "                v0 = l0 ? a : v0;  u0 = l0 ? c : u0;\n"
    return ("""
    uint sg = thread_position_in_grid.x / 32, lane = thread_position_in_grid.x % 32;
    uint rbase = sg * 16, xrow = (row0[0] + rbase) * D;
    const uint DK = DP / 8, DFULL = D / 8;
    threadgroup float tile[SGPG * 256];
    threadgroup float *T = tile + (thread_position_in_threadgroup.x / 32) * 256;
    simdgroup_float8x8 a0[DK], a1[DK];
    for (uint kk = 0; kk < DFULL; kk++) {
        simdgroup_load(a0[kk], X + xrow + kk * 8, D);
        simdgroup_load(a1[kk], X + xrow + 8 * D + kk * 8, D);
    }
    if (DFULL < DK) {                              // ragged last tile: stage it zero-padded on chip
        for (uint e = lane; e < 128; e += 32) {
            uint r = e / 8, j = DFULL * 8 + (e % 8);
            T[e] = (j < D) ? X[xrow + r * D + j] : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(a0[DFULL], T, 8);
        simdgroup_load(a1[DFULL], T + 64, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint i = rbase + lane, xi = (row0[0] + i) * D;
    float xsq = 0;
    if (lane < 16) for (uint j = 0; j < D; j++) xsq += X[xi + j] * X[xi + j];
    float g3 = 3 * gamma[0];
""" + "".join(f"    float v{j} = INFINITY; uint u{j} = 0;\n" for j in range(r)) + """
    for (uint ct = 0; ct < KP / 16; ct++) {
        uint c0 = ct * 16;
        simdgroup_float8x8 b0, b1;
        simdgroup_float8x8 acc00 = simdgroup_float8x8(0.0f), acc01 = simdgroup_float8x8(0.0f);
        simdgroup_float8x8 acc10 = simdgroup_float8x8(0.0f), acc11 = simdgroup_float8x8(0.0f);
        for (uint kk = 0; kk < DK; kk++) {
            simdgroup_load(b0, Ct + kk * 8 * KP + c0, KP);
            simdgroup_load(b1, Ct + kk * 8 * KP + c0 + 8, KP);
            simdgroup_multiply_accumulate(acc00, a0[kk], b0, acc00);
            simdgroup_multiply_accumulate(acc01, a0[kk], b1, acc01);
            simdgroup_multiply_accumulate(acc10, a1[kk], b0, acc10);
            simdgroup_multiply_accumulate(acc11, a1[kk], b1, acc11);
        }
        simdgroup_store(acc00, T, 16);
        simdgroup_store(acc01, T + 8, 16);
        simdgroup_store(acc10, T + 8 * 16, 16);
        simdgroup_store(acc11, T + 8 * 16 + 8, 16);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 16) {
            for (uint t = 0; t < 16; t++) {
                uint c = c0 + t;
                float a = csq[c] - 2 * T[lane * 16 + t];
""" + keep + """
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
""" + f"    bool wide = (lane < 16) && (v{r-1} <= v0 + g3 * (xsq + csq[u0]) + g3 * (xsq + csqmax[0]));\n" + """
    float any = simd_max(wide ? 1.0f : 0.0f);      // the whole simdgroup pays for one widened row
    if (lane == 0) fb[sg] = (uint)any;
    if (lane >= 16) return;
    float best = INFINITY; uint bl = 0;
    if (wide) {
        for (uint c = 0; c < K; c++) {
            float s = 0;
            for (uint j = 0; j < D; j++) { float q = X[xi + j] - C[c * D + j]; s += q * q; }
            bool lt = s < best; best = select(best, s, lt); bl = select(bl, c, lt);
        }
    } else {
""" + "".join(f"        SCORE(u{j})\n" for j in range(r)) + """
    }
    labels[i] = bl; best_d[i] = best;
""")


def _tiles1_reload_src(r):
    """The same kernel with the row tiles re-read from X inside the centre loop, not held in registers.

    Holding them across the whole centre loop is what caps the register version at 96 dims. Re-reading
    costs one extra pair of simdgroup_loads per centre tile, but a 16 x dims row block is only kilobytes
    and stays in cache, which this machine serves at ~2 TB/s against 400-480 GB/s for DRAM - so it is
    nearly free, and the cap disappears. Measured against the materialising tiles path: 1.49x at 128 dims,
    1.30x at 160, 1.29x at 192, then 0.87x at 256, where the re-reads finally cost more than the n x k
    matrix they avoid.

    Staging the block in threadgroup memory instead was tried and is worse - 0.78x at 192 dims, 0.31x at
    480 - because the register file is far larger than the 32 KB of threadgroup memory, so that trades a
    plentiful resource for a scarce one and occupancy collapses as dimension grows.

    Needs dims to be a multiple of 8: there is nowhere to stage a ragged tile once per centre tile.
    """
    body = _tiles1_src(r)
    body = body.replace('    simdgroup_float8x8 a0[DK], a1[DK];\n    for (uint kk = 0; kk < DFULL; kk++) {\n        simdgroup_load(a0[kk], X + xrow + kk * 8, D);\n        simdgroup_load(a1[kk], X + xrow + 8 * D + kk * 8, D);\n    }\n    if (DFULL < DK) {                              // ragged last tile: stage it zero-padded on chip\n        for (uint e = lane; e < 128; e += 32) {\n            uint r = e / 8, j = DFULL * 8 + (e % 8);\n            T[e] = (j < D) ? X[xrow + r * D + j] : 0.0f;\n        }\n        simdgroup_barrier(mem_flags::mem_threadgroup);\n        simdgroup_load(a0[DFULL], T, 8);\n        simdgroup_load(a1[DFULL], T + 64, 8);\n        simdgroup_barrier(mem_flags::mem_threadgroup);\n    }\n', "")
    body = body.replace('        for (uint kk = 0; kk < DK; kk++) {\n            simdgroup_load(b0, Ct + kk * 8 * KP + c0, KP);', '        for (uint kk = 0; kk < DK; kk++) {\n            simdgroup_load(b0, Ct + kk * 8 * KP + c0, KP);'.replace(
        "            simdgroup_load(b0,",
        "            simdgroup_load(ar0, X + xrow + kk * 8, D);\n"
        "            simdgroup_load(ar1, X + xrow + 8 * D + kk * 8, D);\n"
        "            simdgroup_load(b0,"))
    body = body.replace("simdgroup_float8x8 b0, b1;", "simdgroup_float8x8 b0, b1, ar0, ar1;")
    return body.replace("a0[kk]", "ar0").replace("a1[kk]", "ar1")


def _tiles1_nearest(x, Cm, n, k, d, flags=None):
    """Nearest centre with no materialised distance matrix -> (labels, best). Exact.

    When `flags` is given the dataset has not been seen before: the first TILES1_PROBE_ROWS rows run on
    their own and their fallback counts are read back, and if they went badly the rest of the slice goes to
    the rows kernel instead of finishing a pass already known to be losing. That read is the one place this
    path waits for the GPU mid-pass, and it happens once per dataset - the caller records the verdict.
    """
    mx = _mx()
    eps = float(np.finfo(np.float32).eps)
    dp, kp = -(-d // 8) * 8, -(-k // 16) * 16
    csq = (Cm * Cm).sum(1)
    csqmax = mx.max(csq).reshape(1)                    # over the real centres, before any padding
    # Padding is exact, not an approximation: the extra dimensions are zero in both the row and the centre,
    # so they contribute +0.0f terms that cannot move a float32 accumulator, and the padded centres carry
    # csq = +inf so they never enter the candidate set. Only the centre matrix is copied - X is untouched,
    # and its own ragged tile is staged on chip inside the kernel.
    Ctp = np.zeros((dp, kp), dtype=np.float32)
    Ctp[:d, :k] = np.array(Cm).T
    Ct = mx.array(Ctp)
    if kp > k:
        csq = mx.concatenate([csq, mx.full((kp - k,), float("inf"), dtype=mx.float32)])
    gamma = mx.array([dp * eps / (1 - dp * eps)], dtype=mx.float32)
    reload = d > TILES1_DIMS[1]
    kern = _kernel(f"kmeans_tiles1_{TILES1_KEEP}_{int(reload)}",
                   ["X", "Ct", "C", "csq", "csqmax", "row0", "gamma"], ["labels", "best_d", "fb"],
                   _tiles1_reload_src(TILES1_KEEP) if reload else _tiles1_src(TILES1_KEEP), _SCORE_HDR)
    whole = n // 16 * 16                     # whole row tiles here; the tail goes to the rows kernel
    labels, best = [], []

    def tiles(r0, m):
        lab, bst, fell = kern(inputs=[x, Ct, Cm, csq, csqmax, _u32(r0), gamma],
                               template=[("D", d), ("DP", dp), ("K", k), ("KP", kp),
                                         ("SGPG", TILES1_SIMDGROUPS)],
                               grid=((m // 16) * 32, 1, 1), threadgroup=(TILES1_SIMDGROUPS * 32, 1, 1),
                               output_shapes=[(m,), (m,), (m // 16,)],
                               output_dtypes=[mx.uint32, mx.float32, mx.uint32])
        labels.append(lab)
        best.append(bst)
        return fell

    done = 0
    if flags is not None and whole:
        done = min(whole, TILES1_PROBE_ROWS // 16 * 16)
        fell = tiles(0, done)
        flags.append(fell)
        if float(mx.sum(fell)) / (done // 16) > TILES1_MAX_FALLBACK and done < n:
            lab, bst = _rows_kernel(x, Cm, done, n - done, k, d)
            labels.append(lab)
            best.append(bst)
            return mx.concatenate(labels), mx.concatenate(best)
    if whole > done:
        tiles(done, whole - done)
    if whole < n:
        lab, bst = _rows_kernel(x, Cm, whole, n - whole, k, d)
        labels.append(lab)
        best.append(bst)
    return (labels[0], best[0]) if len(labels) == 1 else (mx.concatenate(labels), mx.concatenate(best))


# Distance from each row to the centre it was already assigned. Used by the approximate path so that the
# inertia it reports is the true inertia of the labels it returned, rather than the approximation that chose
# them - otherwise an approximate run could not be compared against an exact one at all. Same accumulation
# order as every other exact distance here.
_LABEL_DIST_SRC = """
    uint i = thread_position_in_grid.x, xi = (row0[0] + i) * D, ci = labels[i] * D;
    float s = 0;
    for (uint j = 0; j < D; j++) { float t = X[xi + j] - C[ci + j]; s += t * t; }
    best_d[i] = s;
"""


# Argmin over a chunk of the product, in one pass. Writing this as MLX ops - csq[None,:] - 2*(x @ Ct) then
# argmin - walks the n x k product four times: the matmul writes it, the scale reads and writes it, the
# subtract reads and writes it, and the argmin reads it again. At 960 dims that intermediate is gigabytes,
# so the arithmetic stops mattering. One simdgroup per row so the 32 lanes read 32 consecutive centres:
# measured 360 GB/s against 257 for a thread per row.
_APPROX_ARGMIN_SRC = """
    uint i = thread_position_in_grid.x / 32, lane = thread_position_in_grid.x % 32, di = i * K;
    float amin = INFINITY; uint cmin = 0;
    for (uint c = lane; c < K; c += 32) {
        float a = csq[c] - 2 * (float)dot[di + c];
        bool lt = a < amin; amin = select(amin, a, lt); cmin = select(cmin, c, lt);
    }
    float g = simd_min(amin);
    uint cc = (amin == g) ? cmin : 0xffffffffu;      // lowest index among those achieving the minimum
    cc = simd_min(cc);
    if (lane == 0) labels[i] = cc;
"""


def _approx_labels(xin, Ct, csq, n, k):
    """Chunked matmul + one fused argmin pass -> labels. xin/Ct may be float32 or float16."""
    mx = _mx()
    per = max(1, DIST_BYTES // (xin.dtype.size * k))     # an fp16 product is half the bytes: half the chunks
    amin = _kernel("kmeans_approx_argmin", ["dot", "csq"], ["labels"], _APPROX_ARGMIN_SRC)
    labels = []
    for r0 in range(0, n, per):
        m = min(per, n - r0)
        lab = amin(inputs=[xin[r0:r0 + m] @ Ct, csq], template=[("K", k)], grid=(m * 32, 1, 1),
                   threadgroup=(256, 1, 1), output_shapes=[(m,)], output_dtypes=[mx.uint32])[0]
        mx.eval(lab)                          # bound memory: one chunk of the product alive at a time
        labels.append(lab)
    return labels[0] if len(labels) == 1 else mx.concatenate(labels)


# The sorted accumulator, but computing each row's distance to its centre on the way through instead of
# taking it as input. A segment is one cluster, so the centre is fixed for the whole segment and cache-
# resident; the row is being read anyway. This removes the separate distance pass the approximate path used
# to make - a whole extra read of X, 4.7 ms of a 41 ms pass on GIST.
_SEGMENT_DIST_SRC = """
    uint s = thread_position_in_grid.x / 32, lid = thread_position_in_grid.x % 32;
    const uint W = D + 2, DPD = (D + 31) / 32;
    uint ci = seg_cluster[s] * D;
    float acc[DPL], carry[DPL];
    for (uint c = 0; c < DPL; c++) { acc[c] = 0.0f; carry[c] = 0.0f; }
    for (uint p = seg_start[s]; p < seg_end[s]; p++) {
        uint r = perm[p], xi = r * D;
        float dd = 0;
        for (uint c = 0; c < DPD; c++) {
            uint j = lid + c * 32;
            if (j < D) { float q = (float)X[xi + j] - C[ci + j]; dd += q * q; }
        }
        dd = simd_sum(dd);
        if (lid == 0) best_out[r] = dd;
        for (uint c = 0; c < DPL; c++) {
            uint e = lid + c * 32;
            if (e < W) {
                float v = (e >= 2) ? (float)X[xi + e - 2] : ((e == 0) ? 1.0f : dd);
                KAHAN_ADD(acc[c], carry[c], v);
            }
        }
    }
    for (uint c = 0; c < DPL; c++) {
        uint e = lid + c * 32;
        if (e < W) { sums[s * W + e] = acc[c]; comp[s * W + e] = carry[c]; }
    }
"""


# Rows grouped by label without a comparison sort. Labels are uint32 below k, so mx.argsort - O(n log n),
# 6 ms at 10M rows - was the wrong tool; it was 22% of the logs step. Three kernels, deterministic and
# stable (rows of one label keep ascending order, which is what the segment kernels' summation order rests
# on): per-block histograms via threadgroup atomics (counts only, so order cannot matter), a scan over
# blocks per label, and a sequential per-block scatter. Measured against mx.argsort: 1.2x at 2M rows k=64,
# 4.7x at 10M k=256, 6.0x at 20M k=32. The per-label totals come out of the scan for free, replacing the
# scatter-add that computed counts before.
_CSORT_HIST_SRC = """
    threadgroup atomic_uint h[K];
    uint lid = thread_position_in_threadgroup.x, b = threadgroup_position_in_grid.x;
    for (uint c = lid; c < K; c += TG) atomic_store_explicit(&h[c], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint start = b * BS, end = min(start + BS, n_rows[0]);
    for (uint i = start + lid; i < end; i += TG) atomic_fetch_add_explicit(&h[labels[i]], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = lid; c < K; c += TG) hist[b * K + c] = atomic_load_explicit(&h[c], memory_order_relaxed);
"""
_CSORT_SCAN_SRC = """
    uint c = thread_position_in_grid.x;
    uint run = 0;
    for (uint b = 0; b < NB; b++) { uint v = hist[b * K + c]; off[b * K + c] = run; run += v; }
    total[c] = run;
"""
_CSORT_SCATTER_SRC = """
    threadgroup uint cnt[TGB * K];
    uint lid = thread_position_in_threadgroup.x, g = threadgroup_position_in_grid.x;
    uint bl = lid, b = g * TGB + bl;                  // TGB blocks per threadgroup, one thread walks each
    for (uint e = lid; e < TGB * K; e += TG) {
        uint bb = g * TGB + e / K, c = e % K;
        cnt[e] = (bb < NB) ? off[bb * K + c] + base[c] : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (bl < TGB && b < NB) {
        uint start = b * BS, end = min(start + BS, n_rows[0]);
        for (uint i = start; i < end; i++) {
            uint c = labels[i];
            uint p = cnt[bl * K + c]; cnt[bl * K + c] = p + 1;
            perm[p] = i;
        }
    }
"""
CSORT_ROWS_PER_BLOCK = 4096
CSORT_MAX_K = 2048                      # the scatter keeps TGB*K counters in threadgroup memory


def _group_by_label(labels, n, k):
    """-> (perm uint32, counts int64): rows ordered by label, ascending within a label, and how many each."""
    mx = _mx()
    if k > CSORT_MAX_K:
        perm = mx.argsort(labels).astype(mx.uint32)
        counts = _host(mx.zeros((k,), dtype=mx.uint32).at[labels].add(mx.array(1, dtype=mx.uint32)))
        return perm, counts.astype(np.int64)
    bs = CSORT_ROWS_PER_BLOCK
    nb = -(-n // bs)
    h = _kernel("kmeans_csort_hist", ["labels", "n_rows"], ["hist"], _CSORT_HIST_SRC)
    hist = h(inputs=[labels, _u32(n)], template=[("K", k), ("BS", bs), ("TG", 256)],
             grid=(nb * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(nb * k,)], output_dtypes=[mx.uint32])[0]
    sc = _kernel("kmeans_csort_scan", ["hist"], ["off", "total"], _CSORT_SCAN_SRC)
    off, total = sc(inputs=[hist], template=[("K", k), ("NB", nb)], grid=(k, 1, 1), threadgroup=(min(k, 256), 1, 1),
                    output_shapes=[(nb * k,), (k,)], output_dtypes=[mx.uint32, mx.uint32])
    base = (mx.cumsum(total) - total).astype(mx.uint32)
    tgb = min(256, max(1, (TG_FLOATS // k)))          # counters must fit the 32 KB of threadgroup memory
    st = _kernel("kmeans_csort_scatter", ["labels", "off", "base", "n_rows"], ["perm"], _CSORT_SCATTER_SRC)
    perm = st(inputs=[labels, off, base, _u32(n)],
              template=[("K", k), ("NB", nb), ("BS", bs), ("TGB", tgb), ("TG", 256)],
              grid=(-(-nb // tgb) * 256, 1, 1), threadgroup=(256, 1, 1),
              output_shapes=[(n,)], output_dtypes=[mx.uint32])[0]
    return perm, _host(total).astype(np.int64)


def _accumulate_sorted_dist(x, labels, k, Cm):
    """Sorted accumulation that also returns each row's distance to its centre -> (totals, best)."""
    mx = _mx()
    n, d = x.shape
    w = d + 2
    perm, counts = _group_by_label(labels, n, k)
    offs = np.concatenate([[0], np.cumsum(counts)])
    nseg = -(-counts // SEGMENT_ROWS)
    seg_first = np.concatenate([[0], np.cumsum(nseg)])
    seg_cluster = np.repeat(np.arange(k), nseg)
    starts = offs[seg_cluster] + (np.arange(len(seg_cluster)) - seg_first[seg_cluster]) * SEGMENT_ROWS
    ends = np.minimum(starts + SEGMENT_ROWS, offs[seg_cluster + 1])
    ns = len(starts)
    seg = _kernel("kmeans_segments_dist", ["X", "C", "perm", "seg_start", "seg_end", "seg_cluster"],
                  ["sums", "comp", "best_out"], _SEGMENT_DIST_SRC, _KAHAN)
    sums, comp, best = seg(inputs=[x, Cm, perm, mx.array(starts.astype(np.uint32)), mx.array(ends.astype(np.uint32)),
                                   mx.array(seg_cluster.astype(np.uint32))],
                           template=[("D", d), ("DPL", -(-w // 32))], grid=(ns * 32, 1, 1), threadgroup=(256, 1, 1),
                           output_shapes=[(ns * w,), (ns * w,), (n,)],
                           output_dtypes=[mx.float32, mx.float32, mx.float32])
    red = _kernel("kmeans_segment_reduce", ["sums", "comp", "seg_first"], ["total", "total_comp"], _SEGMENT_REDUCE_SRC, _KAHAN)
    total, total_comp = red(inputs=[sums, comp, mx.array(seg_first.astype(np.uint32))], template=[("W", w)],
                            grid=(w, k, 1), threadgroup=(32, 8, 1),
                            output_shapes=[(k * w,), (k * w,)], output_dtypes=[mx.float32, mx.float32])
    tot = (_host(total, np.float64) + _host(total_comp, np.float64)).reshape(k, w)
    return tot, best


# float16 for the approximate multiply: MLX's matmul runs at ~49 TFLOP/s in fp16 against ~34 in fp32 on GIST.
# X is converted ONCE per dataset and kept (converting per call costs 5.5 ms and eats the gain - an earlier
# test did exactly that and wrongly concluded fp16 was worth only 6%). fp16 tops out at 65504, so the
# product x.c must stay well inside it: |x.c| <= |x||c| (Cauchy-Schwarz), and the guard requires that bound
# under a quarter of the range. SIFT fails it (|x|^2 ~ 2.6e5, every row overflows, 99.9% of labels garbage);
# GIST passes with room to spare (bound ~55). Accumulation always reads the original float32 X.
APPROX_MIN_DIMS_FP16 = 64               # ...but when the data admits the fp16 multiply the crossover is much
                                        # lower: measured approx ahead from 64 dims (1.14x at k=64 and 256,
                                        # 1.01x at 1024; 1.26x at 80; 1.34x at 96) and behind at 48 and below.
                                        # On data that refuses fp16 - SIFT - the cascade still wins at 128
                                        # (21.1 against 24.4 ms), so the 256 threshold stands there.
APPROX_FP16_LIMIT = 65504.0 / 4
APPROX_FP16_ACCUMULATE = True           # sum the new centres from the fp16 copy too (sums stay float32, compensated)
_approx_state = {}


def _approx_prepare(parts):
    """-> per-dataset state: max |x|^2 (computed once), float16 copies (built on first use, kept)."""
    mx = _mx()
    key = tuple(p.shape for p in parts)
    st = _approx_state.get(key)
    if st is not None and not _holds(st, parts):
        st = None                                       # same shape, different data
    if st is None:
        _approx_state.clear()                           # one dataset at a time; the copies are large
        st = _hold(_approx_state, key, {"half": None,
                                        "xsqmax": max(float(mx.max(mx.sum(p * p, axis=1))) for p in parts)}, parts)
    return st


def _approx_fp16_ok(st, C):
    """Cauchy-Schwarz: |x.c| <= |x||c| must sit well inside float16's range for the product to be usable."""
    return np.isfinite(st["xsqmax"]) and \
        np.sqrt(st["xsqmax"] * float(np.einsum("ij,ij->i", C, C).max())) < APPROX_FP16_LIMIT


def _approx_nearest(x, Cm, n, k, d):
    """Nearest centre from matmul-derived distances, taken on trust -> (labels, exact best).

    NOT exact, and not the default: this is the expanded form |c|^2 - 2 x.c with no verification, which is
    what a k-means written directly on MLX does. It is worth having because above about 320 dims MLX's
    matmul reaches hardware this library's own kernels cannot - 28.7 against a 15.7 TFLOP/s ceiling for
    simdgroup_multiply_accumulate - at about 630x the error, which is far too much for the exact path's
    candidate bound but may be irrelevant to the application. Below that crossover the exact kernels are
    faster anyway, so `exact=False` does not use this there.
    """
    mx = _mx()
    csq = (Cm * Cm).sum(1)
    Ct = mx.contiguous(Cm.T)
    per = max(1, DIST_BYTES // (4 * k))
    amin = _kernel("kmeans_approx_argmin", ["dot", "csq"], ["labels"], _APPROX_ARGMIN_SRC)
    labels = []
    for r0 in range(0, n, per):
        m = min(per, n - r0)
        dot = x[r0:r0 + m] @ Ct
        lab = amin(inputs=[dot, csq], template=[("K", k)], grid=(m * 32, 1, 1), threadgroup=(256, 1, 1),
                   output_shapes=[(m,)], output_dtypes=[mx.uint32])[0]
        mx.eval(lab)                          # bound memory: one chunk of the product alive at a time
        labels.append(lab)
    lab = labels[0] if len(labels) == 1 else mx.concatenate(labels)
    kern = _kernel("kmeans_label_dist", ["X", "C", "labels", "row0"], ["best_d"], _LABEL_DIST_SRC)
    best = kern(inputs=[x, Cm, lab, _u32(0)], template=[("D", d)], grid=(n, 1, 1), threadgroup=(64, 1, 1),
                output_shapes=[(n,)], output_dtypes=[mx.float32])[0]
    return lab, best


# Two-stage exact assignment for dimensions the one-pass path cannot reach.
#
# A partial sum over the first m dimensions lower-bounds the full squared distance, because the remaining
# terms are non-negative. So if that partial already exceeds a distance the row has ACHIEVED - the centre it
# held last iteration, scored exactly - the centre cannot be the nearest, and it never has to be finished.
# Stage 1 runs the same fused tile multiply as the one-pass path but over only those m dims, and writes one
# bit per centre. Stage 2 walks the set bits and scores those centres exactly. Nothing approximate survives
# into the answer: the bound decides only what to skip.
#
# Which m dims matters enormously. Ordered by variance, 4.2% of centres survive on SIFT1M at 128 dims with
# m=64; in the order given, 11.4%. On data with no variance structure at all - isotropic Gaussians - a half
# prefix is half of every distance and nothing prunes, so the kernel reports how many survived and the
# dataset drops back to the ordinary path.
#
# Measured on SIFT1M with a genuinely stale bound (6.9% of labels out of date), zero wrong labels:
#     1M x 128 k=1024, m=64   33.5 -> 18.0 ms   1.86x      1M x 128 k=4096, m=64  132.6 -> 57.2 ms  2.32x
_CASCADE_MASK_SRC = """
    uint sg = thread_position_in_grid.x / 32, lane = thread_position_in_grid.x % 32;
    uint rbase = sg * 16, xrow = (row0[0] + rbase) * M;
    const uint MK = MP / 8, MFULL = M / 8;
    threadgroup float tile[SGPG * 256];
    threadgroup float *T = tile + (thread_position_in_threadgroup.x / 32) * 256;
    simdgroup_float8x8 a0[MK], a1[MK];
    for (uint kk = 0; kk < MFULL; kk++) {
        simdgroup_load(a0[kk], Xm + xrow + kk * 8, M);
        simdgroup_load(a1[kk], Xm + xrow + 8 * M + kk * 8, M);
    }
    if (MFULL < MK) {
        for (uint e = lane; e < 128; e += 32) {
            uint r = e / 8, j = MFULL * 8 + (e % 8);
            T[e] = (j < M) ? Xm[xrow + r * M + j] : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_load(a0[MFULL], T, 8);
        simdgroup_load(a1[MFULL], T + 64, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint i = rbase + lane, xi = (row0[0] + i) * M;
    float xsq = 0;
    if (lane < 16) for (uint j = 0; j < M; j++) xsq += Xm[xi + j] * Xm[xi + j];
    float g3 = 3 * gamma[0];
    float u = (lane < 16) ? ub[i] : 0.0f;
    uint word = 0, nkeep = 0;
    for (uint ct = 0; ct < KP / 16; ct++) {
        uint c0 = ct * 16;
        simdgroup_float8x8 b0, b1;
        simdgroup_float8x8 acc00 = simdgroup_float8x8(0.0f), acc01 = simdgroup_float8x8(0.0f);
        simdgroup_float8x8 acc10 = simdgroup_float8x8(0.0f), acc11 = simdgroup_float8x8(0.0f);
        for (uint kk = 0; kk < MK; kk++) {
            simdgroup_load(b0, Ct + kk * 8 * KP + c0, KP);
            simdgroup_load(b1, Ct + kk * 8 * KP + c0 + 8, KP);
            simdgroup_multiply_accumulate(acc00, a0[kk], b0, acc00);
            simdgroup_multiply_accumulate(acc01, a0[kk], b1, acc01);
            simdgroup_multiply_accumulate(acc10, a1[kk], b0, acc10);
            simdgroup_multiply_accumulate(acc11, a1[kk], b1, acc11);
        }
        simdgroup_store(acc00, T, 16);
        simdgroup_store(acc01, T + 8, 16);
        simdgroup_store(acc10, T + 8 * 16, 16);
        simdgroup_store(acc11, T + 8 * 16 + 8, 16);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < 16) {
            uint bits = 0;
            for (uint t = 0; t < 16; t++) {
                uint c = c0 + t;
                float a = xsq + csq[c] - 2 * T[lane * 16 + t];
                // c >= K are padding, and carry csq = +inf so the bound below is inf - inf = NaN; NaN fails
                // every comparison, so without this guard they would all survive and be scored out of range
                if (c < K && !(a - g3 * (xsq + csq[c]) > u)) { bits |= (1u << t); nkeep++; }
            }
            word |= bits << ((ct & 1u) * 16);
            if (ct & 1u) { mask[i * (KP / 32) + (ct >> 1)] = word; word = 0; }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane < 16) nsurv[i] = nkeep;
"""

# Stage 2: walk the surviving bits and score those centres exactly. The row is held once across the lanes,
# so X is read once rather than once per candidate, and both reads coalesce.
_CASCADE_SCORE_SRC = """
    uint i = thread_position_in_grid.x / 32, lane = thread_position_in_grid.x % 32;
    const uint DPL = (D + 31) / 32, W = KP / 32;
    float xv[DPL];
    for (uint c = 0; c < DPL; c++) { uint j = lane + c * 32; xv[c] = (j < D) ? X[(row0[0] + i) * D + j] : 0.0f; }
    float best = INFINITY; uint bl = 0;
    for (uint w = 0; w < W; w++) {
        uint bits = mask[i * W + w];
        while (bits) {
            uint t = ctz(bits);
            bits &= bits - 1;
            uint cc = w * 32 + t, ci = cc * D;
            float s = 0;
            for (uint c = 0; c < DPL; c++) {
                uint j = lane + c * 32;
                float q = xv[c] - ((j < D) ? C[ci + j] : 0.0f);
                s += q * q;
            }
            s = simd_sum(s);
            bool lt = (s < best) || (s == best && cc < bl);
            best = select(best, s, lt); bl = select(bl, cc, lt);
        }
    }
    if (lane == 0) { labels[i] = bl; best_d[i] = best; }
"""


def _cascade_m(d):
    """Prefix width to prune on: half the dimensions, capped where the tiles stop fitting in registers."""
    return min(TILES1_DIMS[1], max(8, (d // 2) // 8 * 8))


def _cascade_ok(d, k):
    return CASCADE_MIN_DIMS <= d <= CASCADE_MAX_DIMS and k >= CASCADE_MIN_K


def _cascade_prefix(x, order, m):
    mx = _mx()
    return mx.contiguous(mx.take(x, mx.array(order[:m].astype(np.uint32)), axis=1))


def _cascade_nearest(x, Cm, C, Xm, order, prev, n, k, d, m):
    """Exact nearest centre by pruning on the prefix -> (labels, best, survivor fraction)."""
    mx = _mx()
    eps = float(np.finfo(np.float32).eps)
    mp, kp = -(-m // 8) * 8, -(-k // 32) * 32
    Cm_pref = np.ascontiguousarray(C[:, order[:m]])
    Ctp = np.zeros((mp, kp), dtype=np.float32)
    Ctp[:m, :k] = Cm_pref.T
    csq = np.full(kp, np.inf, dtype=np.float32)          # padded centres never survive the bound
    csq[:k] = (Cm_pref.astype(np.float64) ** 2).sum(1)
    gamma = mx.array([mp * eps / (1 - mp * eps)], dtype=mx.float32)
    dist = _kernel("kmeans_label_dist", ["X", "C", "labels", "row0"], ["best_d"], _LABEL_DIST_SRC)
    whole = n // 16 * 16
    labels, best = [], []
    surv = 0.0
    if whole:
        ub = dist(inputs=[x, Cm, prev, _u32(0)], template=[("D", d)], grid=(whole, 1, 1),
                  threadgroup=(64, 1, 1), output_shapes=[(whole,)], output_dtypes=[mx.float32])[0]
        k1 = _kernel(f"kmeans_cascade_mask_{m}", ["Xm", "Ct", "csq", "ub", "row0", "gamma"],
                     ["mask", "nsurv"], _CASCADE_MASK_SRC)
        mask, nsurv = k1(inputs=[Xm, mx.array(Ctp), mx.array(csq), ub, _u32(0), gamma],
                         template=[("M", m), ("MP", mp), ("K", k), ("KP", kp),
                                   ("SGPG", TILES1_SIMDGROUPS)],
                         grid=((whole // 16) * 32, 1, 1), threadgroup=(TILES1_SIMDGROUPS * 32, 1, 1),
                         output_shapes=[(whole * (kp // 32),), (whole,)],
                         output_dtypes=[mx.uint32, mx.uint32])
        k2 = _kernel(f"kmeans_cascade_score_{d}", ["X", "C", "mask", "row0"], ["labels", "best_d"],
                     _CASCADE_SCORE_SRC)
        lab, bst = k2(inputs=[x, Cm, mask, _u32(0)], template=[("D", d), ("KP", kp)],
                      grid=(whole * 32, 1, 1), threadgroup=(256, 1, 1),
                      output_shapes=[(whole,), (whole,)], output_dtypes=[mx.uint32, mx.float32])
        labels.append(lab)
        best.append(bst)
        surv = float(mx.sum(nsurv)) / (whole * k)
    if whole < n:
        lab, bst = _rows_kernel(x, Cm, whole, n - whole, k, d)
        labels.append(lab)
        best.append(bst)
    out = (labels[0], best[0]) if len(labels) == 1 else (mx.concatenate(labels), mx.concatenate(best))
    return out[0], out[1], surv


def _nearest(x, Cm, method, fb_out=None, lb_out=None):
    """Nearest center and its distance for every row of one slice -> (labels uint32, best float32)."""
    mx = _mx()
    n, d = x.shape
    k = Cm.shape[0]
    if method == "approx":
        return _approx_nearest(x, Cm, n, k, d)
    if method == "tiles1":
        return _tiles1_nearest(x, Cm, n, k, d, fb_out)
    if method == "tiles":
        return _tiles_nearest(x, Cm, n, k, d, lb_out)
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
        # Both remaining choices are deterministic and compensated. "blocks" is no longer reachable from here:
        # once the segment kernel stopped read-modify-writing device memory, sorted beat it everywhere
        # measured, by 1.4x at 200k rows and 4133 ms to 9 ms on GIST1M. It stays available by name.
        fits = 2 * k * w <= TG_FLOATS
        small = k * w <= LANES_SMALL_KW or n >= LANES_ROWS_PER_SLOT * k * w
        method = "lanes" if fits and small else "sorted"
    if method == "lanes":
        return _accumulate_lanes(x, labels, best, k)
    if method == "atomic":
        return _accumulate_atomic(x, labels, best, k)
    return _accumulate_sorted(x, labels, best, k) if method == "sorted" else _accumulate_blocks(x, labels, best, k)


# Accumulation with the lanes spread across dimensions instead of rows.
#
# The other on-chip kernel (_ATOMIC_SRC) gives each lane a row, so all 32 lanes of a simdgroup race for the
# same cluster slot and every add has to be atomic - which costs 1.8x here and, worse, fixes no summation
# order, so it cannot be the default. Give each lane a DIMENSION and the 32 lanes of one row write 32
# different addresses: no lane ever collides with another, no atomics, and rows are consumed in increasing
# index order, so the result is the same on every run. Arithmetic throughput is unchanged - 32 floats per
# instruction slot either way - and because each tile entry has exactly one writer, the Kahan carry can live
# on chip beside it.
#
# One row at a time per simdgroup is latency-bound (the store address depends on a load of labels[i]), so
# ROWS_AHEAD rows are loaded together and then added one at a time, in order: the loads overlap, the adds
# stay sequential. Measured at 2M x 50, k=64: 4.8 ms unrolled once, 2.2 ms at 2, 3.0 ms at 8 with
# compensation, against 4.6 ms for the sorted path - and 1.1e-16 against float64, where the sorted path is
# 2.8e-14.
#
# The limit is threadgroup memory: two tiles of k*(dims+2) floats. Packing several simdgroups into one
# threadgroup does not help, because N private tiles cost a core exactly what N threadgroups do (measured
# 1.39 ms against 1.08 ms at k=32 - it is slower, not faster).
_LANES_SRC_HEAD = """
    threadgroup float tile[KW];
    threadgroup float comp[KW];
    uint lid = thread_position_in_threadgroup.x, b = threadgroup_position_in_grid.x, W = D + 2;
    for (uint e = lid; e < KW; e += 32) { tile[e] = 0.0f; comp[e] = 0.0f; }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    uint start = min(b * BS, n_rows[0]), end = min(start + BS, n_rows[0]);
"""
_LANES_SRC_TAIL = """
    for (uint i = stop; i < end; i++) {
        uint o = labels[i] * W, xi = i * D;
        if (lid == 0) { KAHAN_ADD(tile[o], comp[o], 1.0f); KAHAN_ADD(tile[o + 1], comp[o + 1], best_d[i]); }
        for (uint j = lid; j < D; j += 32) KAHAN_ADD(tile[o + 2 + j], comp[o + 2 + j], X[xi + j]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lid; e < KW; e += 32) { sums[b * KW + e] = tile[e]; comp_out[b * KW + e] = comp[e]; }
"""


def _lanes_src(u, dpl):
    """Source for the dimension-per-lane accumulator, unrolled over u rows and dpl dims per lane."""
    decl = "".join(f"    uint o{t}; float bd{t};\n" for t in range(u))
    decl += "".join(f"    float v{t}_{c};\n" for t in range(u) for c in range(dpl))
    load = "".join(f"        o{t} = labels[i + {t}] * W;  bd{t} = best_d[i + {t}];\n" for t in range(u))
    load += "".join(f"        v{t}_{c} = X[(i + {t}) * D + lid + {c} * 32];\n"
                    for t in range(u) for c in range(dpl))
    add = ""
    for t in range(u):
        add += (f"        if (lid == 0) {{ KAHAN_ADD(tile[o{t}], comp[o{t}], 1.0f);"
                f" KAHAN_ADD(tile[o{t} + 1], comp[o{t} + 1], bd{t}); }}\n")
        for c in range(dpl):
            slot = f"o{t} + 2 + lid + {c} * 32"
            add += (f"        if (lid + {c} * 32 < D) "
                    f"KAHAN_ADD(tile[{slot}], comp[{slot}], v{t}_{c});\n")
    return (_LANES_SRC_HEAD + decl
            + f"    uint stop = start + (end - start) / {u} * {u};\n"
            + f"    for (uint i = start; i < stop; i += {u}) {{\n" + load + add + "    }\n"
            + _LANES_SRC_TAIL)


def _accumulate_lanes(x, labels, best, k):
    mx = _mx()
    n, d = x.shape
    w, kw = d + 2, k * (d + 2)
    nb = max(1, min(LANES_BLOCKS, -(-n // LANES_MIN_ROWS)))
    bs = -(-n // nb)
    nb = -(-n // bs)
    dpl = -(-d // 32)
    acc = _kernel(f"kmeans_lanes_{dpl}", ["X", "labels", "best_d", "n_rows"], ["sums", "comp_out"],
                  _lanes_src(LANES_ROWS_AHEAD, dpl), _KAHAN)
    sums, comp = acc(inputs=[x, labels, best, _u32(n)], template=[("D", d), ("KW", kw), ("BS", bs)],
                     grid=(nb * 32, 1, 1), threadgroup=(32, 1, 1),
                     output_shapes=[(nb * kw,), (nb * kw,)], output_dtypes=[mx.float32, mx.float32])
    red = _kernel("kmeans_reduce", ["sums", "comp", "n_blocks"], ["total", "total_comp"], _REDUCE_SRC, _KAHAN)
    total, total_comp = red(inputs=[sums, comp, _u32(nb)], template=[("KW", kw)],
                            grid=(kw, 1, 1), threadgroup=(64, 1, 1),
                            output_shapes=[(kw,), (kw,)], output_dtypes=[mx.float32, mx.float32])
    return (_host(total, np.float64) + _host(total_comp, np.float64)).reshape(k, w)


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
    return _host(sums, np.float64).reshape(nb, k, w).sum(0)


def _accumulate_sorted(x, labels, best, k):
    mx = _mx()
    n, d = x.shape
    w = d + 2
    perm, counts = _group_by_label(labels, n, k)
    offs = np.concatenate([[0], np.cumsum(counts)])
    nseg = -(-counts // SEGMENT_ROWS)                      # segments per cluster (0 for an empty cluster)
    seg_first = np.concatenate([[0], np.cumsum(nseg)])
    seg_cluster = np.repeat(np.arange(k), nseg)
    starts = offs[seg_cluster] + (np.arange(len(seg_cluster)) - seg_first[seg_cluster]) * SEGMENT_ROWS
    ends = np.minimum(starts + SEGMENT_ROWS, offs[seg_cluster + 1])
    ns = len(starts)
    seg = _kernel("kmeans_segments", ["X", "best_d", "perm", "seg_start", "seg_end"], ["sums", "comp"], _SEGMENT_SRC, _KAHAN)
    sums, comp = seg(inputs=[x, best, perm, mx.array(starts.astype(np.uint32)), mx.array(ends.astype(np.uint32))],
                     template=[("D", d), ("DPL", -(-w // 32))], grid=(ns * 32, 1, 1), threadgroup=(256, 1, 1),
                     output_shapes=[(ns * w,), (ns * w,)], output_dtypes=[mx.float32, mx.float32])
    red = _kernel("kmeans_segment_reduce", ["sums", "comp", "seg_first"], ["total", "total_comp"], _SEGMENT_REDUCE_SRC, _KAHAN)
    total, total_comp = red(inputs=[sums, comp, mx.array(seg_first.astype(np.uint32))], template=[("W", w)],
                            grid=(w, k, 1), threadgroup=(32, 8, 1),
                            output_shapes=[(k * w,), (k * w,)], output_dtypes=[mx.float32, mx.float32])
    return (_host(total, np.float64) + _host(total_comp, np.float64)).reshape(k, w)


def _accumulate_blocks(x, labels, best, k):
    mx = _mx()
    n, d = x.shape
    w = d + 2
    nb = min(max(-(-n // ACC_ROWS_PER_BLOCK), ACC_MIN_BLOCKS), ACC_MAX_BLOCKS)   # ~256 rows per thread
    nb = max(-(-n // 2**24), min(nb, max(1, ACC_BYTES // (8 * k * w)), n))       # memory cap; counts stay exact
    bs = -(-n // nb)
    nb = -(-n // bs)
    acc = _kernel("kmeans_accumulate", ["X", "labels", "best_d", "n_rows"], ["sums", "comp"], _ACCUMULATE_SRC, _KAHAN)
    sums, comp = acc(inputs=[x, labels, best, _u32(n)], template=[("D", d), ("K", k), ("NB", nb)],
                     grid=(nb, 1, 1), threadgroup=(min(nb, 64), 1, 1), init_value=0,
                     output_shapes=[(nb * k * w,), (nb * k * w,)], output_dtypes=[mx.float32, mx.float32])
    red = _kernel("kmeans_reduce", ["sums", "comp", "n_blocks"], ["total", "total_comp"], _REDUCE_SRC, _KAHAN)
    total, total_comp = red(inputs=[sums, comp, _u32(nb)], template=[("KW", k * w)],
                            grid=(k * w, 1, 1), threadgroup=(64, 1, 1),
                            output_shapes=[(k * w,), (k * w,)], output_dtypes=[mx.float32, mx.float32])
    return (_host(total, np.float64) + _host(total_comp, np.float64)).reshape(k, w)


# Assigning and accumulating in one kernel. The two-kernel form reads X twice and every row does a
# read-modify-write into a device-memory buffer; both disappear if each thread keeps its own compensated
# slots in threadgroup memory. No atomics, so it stays deterministic, and labels are identical.
#
# The cost is capacity: a thread needs k*(dims+2) slots in each of two arrays, so the threads per group are
# TG_FLOATS/2 divided by that. Measured on 10M rows, gain against a full lloyd_step by k*(dims+2):
# 16-32 -> ~6x, 64 -> 2.4-2.9x, 96 -> 1.85x, 128 -> 1.1-1.3x, 256 -> 0.79x, 448 -> 0.57x. Past ~128 the
# group is too thin to fill the GPU and the per-group zeroing, which does not shrink with the work, takes
# over. Row count, label skew (98% in one cluster still gave 2.97x) and block size barely matter.
_FUSED_SRC = """
    threadgroup float sums[TG * KW];
    threadgroup float comp[TG * KW];
    uint lid = thread_position_in_threadgroup.x, b = threadgroup_position_in_grid.x, W = D + 2;
    for (uint e = 0; e < KW; e++) { sums[lid * KW + e] = 0.0f; comp[lid * KW + e] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint start = b * BS, end = min(start + BS, uint(N));
    for (uint i = start + lid; i < end; i += TG) {
        uint xi = i * D;
        float xv[D];
        for (uint j = 0; j < D; j++) xv[j] = X[xi + j];
        float best = INFINITY; uint bl = 0;
        for (uint c = 0; c < K; c++) {
            float s = 0;
            for (uint j = 0; j < D; j++) { float t = xv[j] - C[c * D + j]; s += t * t; }
            bool lt = s < best; best = select(best, s, lt); bl = select(bl, c, lt);
        }
        labels[i] = bl; best_d[i] = best;
        uint o = lid * KW + bl * W;
        sums[o] += 1.0f;                                   // exact: counts stay well inside 2^24
        KAHAN_ADD(sums[o + 1], comp[o + 1], best);
        for (uint j = 0; j < D; j++) KAHAN_ADD(sums[o + 2 + j], comp[o + 2 + j], xv[j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = lid; e < KW; e += TG) {
        float s = 0.0f, c = 0.0f;
        for (uint t = 0; t < TG; t++) { KAHAN_ADD(s, c, sums[t * KW + e]); KAHAN_ADD(s, c, comp[t * KW + e]); }
        out[b * KW + e] = s + c;
    }
"""
_fused_broken = set()          # (dims, k) this GPU would not compile; probed by trying, once each


def _fused_plan(d, k):
    """-> (threads per group, rows per block) if one kernel can do this shape here, else None."""
    kw = k * (d + 2)
    if kw > FUSED_MAX_KW or d > ROWS_REG_MAX_DIMS or (d, k) in _fused_broken:
        return None
    tg = min(64, (TG_FLOATS // 2 // kw) // 8 * 8)
    return (tg, FUSED_ROWS_PER_BLOCK) if tg >= 8 else None


def _fused_pass(x, Cm, k, tg, bs):
    """Assignment and accumulation in one read of X -> (totals (k, d+2) float64, labels, best)."""
    mx = _mx()
    n, d = x.shape
    w, kw = d + 2, k * (d + 2)
    nb = -(-n // bs)
    kern = _kernel(f"kmeans_fused_{d}_{k}_{tg}", ["X", "C"], ["out", "labels", "best_d"], _FUSED_SRC, _KAHAN)
    out, labels, best = kern(inputs=[x, Cm],
                             template=[("D", d), ("K", k), ("KW", kw), ("TG", tg), ("BS", bs), ("N", n)],
                             grid=(nb * tg, 1, 1), threadgroup=(tg, 1, 1), init_value=0,
                             output_shapes=[(nb * kw,), (n,), (n,)],
                             output_dtypes=[mx.float32, mx.uint32, mx.float32])
    mx.eval(labels, best)
    tot = _host(out, np.float64).reshape(nb, k, w).sum(0)
    return tot, labels, best


_tiles1_slow = {}          # datasets whose error window is too wide for the one-pass path to pay off


def _tiles1_ok(d, k):
    """Shapes where the one-pass tiles path is faster (measured; see the kernel's comment).

    Dimensions and k need not be multiples of anything - the kernel pads both, exactly.
    """
    top = TILES1_RELOAD_DIMS if d % 8 == 0 else TILES1_DIMS[1]
    return TILES1_DIMS[0] <= d <= top and k >= TILES1_MIN_K


# 5. Per-centre distance bounds between iterations (Elkan 2003), for the exact path at high dims.
#
# After the first iteration or two, almost none of the n x k distances a Lloyd pass computes can change an
# assignment: measured on GIST 300k x 960, k=1024, the triangle inequality rules out 97.5% of the pairs at
# iteration 1 and 99.3% from iteration 2 (benchmarks/NOTES.md, "The per-pass benchmark hides the largest lever
# left"). Group bounds (Yinyang) are far weaker on the same data - a group inherits its biggest mover, and
# drift is heavy-tailed - so the bounds are kept per centre: one float32 lower bound per (row, centre), n x k
# in all, which is what limits where the path is offered (BOUNDS_MAX_BYTES).
#
# State per dataset: labels, the exact distance to the assigned centre (ub), the centres those were computed
# against, and - only while they pay - the lower bounds (lb, in row chunks). A pass with new centres C'
# subtracts each centre's drift |c' - c| from its column of lb, recomputes every row's own distance exactly
# (so ub is tight and the inertia exact), and visits only the centres whose bound fell below it. Labels stay
# exact: a centre that is not visited provably cannot be nearer. What differs from the full pass is the tie
# rule - a row keeps its label unless a visited centre is strictly nearer in the computed distance, and an
# exact tie with an unvisited centre cannot be seen - so on constructed ties the two paths can disagree; on
# real data that set has measure zero, and either label is within float32 of optimal.
#
# When the bounds pay is a property of the data, and on isotropic high-dimensional data they never do (the
# margins between centres are a few percent of the distance, smaller than the drift), so a full pass never
# builds them on speculation: building n x k cost as much as the pass it was meant to save. Instead every
# full pass records labels, ub and C (cheap), and the next full pass first asks a strided sample of rows,
# given fresh bounds against the recorded centres from the exact dot kernel, what fraction of the pairs a
# step would visit; only if that is under BOUNDS_MAX_VISITED does the pass build the bounds for every row -
# one extra kernel over the dot chunks it computes anyway - and the step after it uses them. Bounds that
# have decayed past the threshold are dropped, not rebuilt blindly, and a build that lasted a step or less
# makes the next question wait (1, 3, 7 iterations), so data the bounds cannot help pays a few percent.
#
# Floating point: computed squared distances carry a relative error up to gamma = D*eps/(1-D*eps); stored
# bounds are scaled by (1 - 2*gamma) and the own distance by (1 + 2*gamma) before any comparison, and drifts
# are rounded up, so every skip is justified in exact arithmetic. Bounds built from dot products use the
# expanded form's bound, 3*gamma*(|x|^2 + |c|^2), the same one the tiles path's candidate test uses.
#
# Not tried yet: fp16 bounds (halves the memory and the traffic; needs round-toward-zero conversion) and
# rewriting only the rows that visited something.
_bounds_state = {}


def _bounds_ok(parts, k, d):
    return (BOUNDS_MIN_DIMS <= d <= BOUNDS_MAX_DIMS and k >= TILES_MIN_K and k % 8 == 0 and d % 8 == 0
            and sum(p.shape[0] for p in parts) * k * 4 + DIST_BYTES <= BOUNDS_MAX_BYTES)


def _bounds_chunk(k):
    return max(1, DIST_BYTES // (4 * k))            # rows per lower-bound chunk: one DIST_BYTES buffer


# One simdgroup per row, lanes striding over the centres: the dot chunk is read once, coalesced, and the
# row's |x|^2 is formed on the way. As a chain of MLX elementwise ops this was seven passes over each 512 MB
# chunk and cost more than the multiply that produced it (a bounds-building pass took 106-125 ms against 57).
_LB_INIT_SRC = """
    uint i = thread_position_in_grid.x / 32, lane = thread_position_in_grid.x % 32;
    uint xi = (row0[0] + i) * D;
    float xs = 0;
    for (uint j = lane; j < D; j += 32) xs += X[xi + j] * X[xi + j];
    xs = simd_sum(xs);
    float g3 = pads[0], lo = pads[1];
    for (uint j = lane; j < K; j += 32) {
        float sq = xs + csq[j];
        float v = sq - 2 * dot[i * K + j] - g3 * sq;
        lb[i * K + j] = sqrt(max(v, 0.0f)) * lo;
    }
"""


def _lb_from_dot(x, r0, dot, csq, m, k, d, gamma):
    """Lower bound on the distance from each of m rows (from r0) to each centre, from expanded-form dot products."""
    mx = _mx()
    kern = _kernel("kmeans_lb_init", ["X", "dot", "csq", "row0", "pads"], ["lb"], _LB_INIT_SRC)
    pads = mx.array([3 * gamma, 1 - 2 * gamma], dtype=mx.float32)
    lb = kern(inputs=[x, dot, csq, _u32(r0), pads], template=[("D", d), ("K", k)], grid=(m * 32, 1, 1),
              threadgroup=(256, 1, 1), output_shapes=[(m * k,)], output_dtypes=[mx.float32])[0]
    mx.eval(lb)
    return lb


# Both kernels hold the row across the 32 lanes of a simdgroup (D/32 values each) and give each lane every
# 32nd centre's bound, so the bound row is read coalesced and a distance is one lane-parallel pass over the
# row plus a simd_sum. The set of centres to visit in a block of 32 is gathered into a mask with simd_sum of
# one bit per lane and walked with ctz, ascending, so computed-equal ties resolve to the lowest index.
_BOUNDS_HEAD = """
    uint i = thread_position_in_grid.x / 32, lane = thread_position_in_grid.x % 32;
    const uint DPL = (D + 31) / 32;
    uint r = row0[0] + STRIDE * i;
    if (r >= row_end[0]) return;
    float xv[DPL];
    for (uint c = 0; c < DPL; c++) { uint j = lane + c * 32; xv[c] = (j < D) ? X[r * D + j] : 0.0f; }
    uint l = labels_in[LOCAL];
    float s = 0;
    for (uint c = 0; c < DPL; c++) { uint j = lane + c * 32; float q = xv[c] - ((j < D) ? C[l * D + j] : 0.0f); s += q * q; }
    s = simd_sum(s);
    float lo = pads[0], hi = pads[1];
    float dl = sqrt(s), ub = dl * hi;
"""

_BOUNDS_COUNT_SRC = _BOUNDS_HEAD.replace("STRIDE", "stride[0]").replace("LOCAL", "STRIDE * i".replace("STRIDE", "stride[0]")) + """
    uint cnt = 0;
    for (uint w = 0; w < K; w += 32) {
        uint j = w + lane;
        float v = (j < K) ? lb_in[(STRIDE * i) * K + j] - drift[j] : INFINITY;
        cnt += (j != l && v < ub) ? 1u : 0u;
    }
    cnt = simd_sum(cnt);
    if (lane == 0) count[i] = cnt;
""".replace("STRIDE", "stride[0]")

_BOUNDS_STEP_SRC = _BOUNDS_HEAD.replace("STRIDE * i", "i").replace("LOCAL", "i") + """
    uint bl = l; float sbest = s;
    uint nvis = 0;
    for (uint w = 0; w < K; w += 32) {
        uint j = w + lane;
        float v = (j < K) ? lb_in[i * K + j] - drift[j] : INFINITY;
        if (j == l) v = dl * lo;
        uint m = simd_sum((j != l && v < ub) ? (1u << lane) : 0u);
        while (m) {
            uint t = ctz(m); m &= m - 1;
            uint cc = w + t, ci = cc * D;
            float s2 = 0;
            for (uint c = 0; c < DPL; c++) { uint jj = lane + c * 32; float q = xv[c] - ((jj < D) ? C[ci + jj] : 0.0f); s2 += q * q; }
            s2 = simd_sum(s2);
            float dcc = sqrt(s2);
            if (lane == t) v = dcc * lo;
            if (s2 < sbest) { sbest = s2; bl = cc; ub = dcc * hi; }
            nvis++;
        }
        if (j < K) lb_out[i * K + j] = v;
    }
    if (lane == 0) { labels[i] = bl; best_d[i] = sbest; ub_out[i] = sqrt(sbest); visited[i] = nvis; }
"""


def _bounds_edges(n, k):
    """Row chunks: whole simdgroup tiles in DIST_BYTES pieces, then the tail rows (fewer than one tile)."""
    step = 16 if k % 16 == 0 else 8
    chunk = max(step, (DIST_BYTES // (4 * k)) // step * step)
    whole = n // step * step
    return list(range(0, whole, chunk)) + [whole] + ([n] if whole < n else []) if whole else [0, n]


def _lb_chunk(x, Cm, Ct, csq, a, b, k, d, gamma):
    """Lower bounds for rows a..b against Cm, from the exact dot kernels.

    MLX's matmul is not accurate enough for this: its float32 product carries ~630x the error of the direct
    sum (the reason the exact path never used it), and a bound padded for the direct sum's error and built
    from it skipped genuinely nearer centres - 1386 wrong labels in a 80k-row test. The tiles path's own
    simdgroup kernel is exact to gamma; the rows that do not fill a tile take the direct-distance kernel.
    """
    mx = _mx()
    m, step = b - a, 16 if k % 16 == 0 else 8
    if m % step == 0:
        dot_k = (_kernel("kmeans_dot_tiles16", ["X", "Ct", "row0"], ["dot"], _DOT_TILES16_SRC) if step == 16
                 else _kernel("kmeans_dot_tiles", ["X", "Ct", "row0"], ["dot"], _DOT_TILES_SRC))
        dot = dot_k(inputs=[x, Ct, _u32(a)], template=[("D", d), ("K", k)],
                    grid=((m // step) * (k // step) * 32, 1, 1), threadgroup=(TILE_THREADGROUP, 1, 1),
                    output_shapes=[(m * k,)], output_dtypes=[mx.float32])[0]
        return _lb_from_dot(x, a, dot, csq, m, k, d, gamma)
    pairs = _kernel("kmeans_pairs", ["X", "C", "row0"], ["dist"], _PAIRS_SRC)
    dist = pairs(inputs=[x, Cm, _u32(a)], template=[("D", d), ("K", k)], grid=(k, m, 1), threadgroup=(16, 16, 1),
                 output_shapes=[(m, k)], output_dtypes=[mx.float32])[0]
    lb = mx.sqrt(dist).reshape(m * k) * (1 - 2 * gamma)
    mx.eval(lb)
    return lb


def _bounds_gamma(d):
    eps = float(np.finfo(np.float32).eps)
    return d * eps / (1 - d * eps)


def _bounds_record(parts, C, nearest, k, d, lbs=None):
    """After a full pass: keep labels, ub and C for the predictor, and the bounds if the pass built them."""
    mx = _mx()
    st = {"C": np.array(C, dtype=np.float32, copy=True), "labels": [], "ub": [], "lb": lbs, "life": 0}
    for (labels, best), x in zip(nearest, parts):
        edges = _bounds_edges(x.shape[0], k)
        st["labels"].append([labels[a:b] for a, b in zip(edges, edges[1:])])
        st["ub"].append([mx.sqrt(best[a:b]) for a, b in zip(edges, edges[1:])])
        mx.eval(*st["labels"][-1], *st["ub"][-1])
    if lbs is not None:
        assert all(len(l) == len(c) for l, c in zip(lbs, st["labels"]))
    return _hold(_bounds_state, (tuple(p.shape for p in parts), k, d), st, parts)


def _bounds_predict(st, parts, C, k, d):
    """-> the fraction of pairs a bounds step would visit if the bounds were built now, from a strided sample."""
    mx = _mx()
    Cm = mx.array(st["C"])
    Ct, csq = mx.array(np.ascontiguousarray(st["C"].T)), (Cm * Cm).sum(1)
    gamma = _bounds_gamma(d)
    drift = mx.array(_bounds_drift(C, st["C"]))
    pads = _bounds_pads(d)
    kern = _kernel("kmeans_bounds_count", ["X", "C", "drift", "labels_in", "lb_in", "row0", "row_end", "stride", "pads"],
                   ["count"], _BOUNDS_COUNT_SRC)
    total, seen = 0.0, 0
    step = 16 if k % 16 == 0 else 8
    for x, labs in zip(parts, st["labels"]):
        n = x.shape[0]
        idx = mx.arange(0, n, max(1, n // BOUNDS_SAMPLE_ROWS), dtype=mx.uint32)
        idx = idx[: max(step, idx.shape[0] // step * step)] if idx.shape[0] >= step else idx
        xs = x[idx]
        lab = mx.concatenate(labs)[idx] if len(labs) > 1 else labs[0][idx]
        ns = xs.shape[0]
        lb = _lb_chunk(xs, Cm, Ct, csq, 0, ns, k, d, gamma)
        cnt = kern(inputs=[xs, C_mx(C), drift, lab, lb, _u32(0), _u32(ns), _u32(1), pads],
                   template=[("D", d), ("K", k)], grid=(ns * 32, 1, 1), threadgroup=(256, 1, 1),
                   output_shapes=[(ns,)], output_dtypes=[mx.uint32])[0]
        total += float(mx.sum(cnt)); seen += ns
    return total / (seen * k)


def _bounds_drift(C, C0):
    """|c' - c| per centre, rounded up to float32."""
    dr = np.sqrt(((C.astype(np.float64) - C0.astype(np.float64)) ** 2).sum(1)).astype(np.float32)
    return np.nextafter(dr, np.float32(np.inf))


def _bounds_pads(d):
    eps = float(np.finfo(np.float32).eps)
    g = 2 * d * eps / (1 - d * eps)
    return _mx().array([1 - g, 1 + g], dtype=_mx().float32)


def _bounds_count(st, parts, C, k, d):
    """-> fraction of the n x k centre distances the step would compute, estimated on strided rows."""
    mx = _mx()
    drift = mx.array(_bounds_drift(C, st["C"]))
    pads = _bounds_pads(d)
    kern = _kernel("kmeans_bounds_count", ["X", "C", "drift", "labels_in", "lb_in", "row0", "row_end", "stride", "pads"],
                   ["count"], _BOUNDS_COUNT_SRC)
    total, seen = 0.0, 0
    for x, labs, lbs in zip(parts, st["labels"], st["lb"]):
        n = x.shape[0]
        stride = max(1, n // BOUNDS_SAMPLE_ROWS)
        r0 = 0
        for lab, lb in zip(labs, lbs):
            m = lab.shape[0]
            ns = -(-m // stride)
            cnt = kern(inputs=[x, C_mx(C), drift, lab, lb, _u32(r0), _u32(r0 + m), _u32(stride), pads],
                       template=[("D", d), ("K", k)], grid=(ns * 32, 1, 1), threadgroup=(256, 1, 1),
                       output_shapes=[(ns,)], output_dtypes=[mx.uint32])[0]
            total += float(mx.sum(cnt)); seen += ns
            r0 += m
    return total / (seen * k)


def C_mx(C):
    return _mx().array(C)


def _bounds_pass(st, parts, C, k, d, accumulate):
    """One assignment pass from the stored bounds -> (totals, nearest), and the state moved to C."""
    mx = _mx()
    Cm = C_mx(C)
    drift = mx.array(_bounds_drift(C, st["C"]))
    pads = _bounds_pads(d)
    kern = _kernel("kmeans_bounds_step", ["X", "C", "drift", "labels_in", "lb_in", "row0", "row_end", "pads"],
                   ["labels", "best_d", "ub_out", "lb_out", "visited"], _BOUNDS_STEP_SRC)
    tot = np.zeros((k, d + 2), dtype=np.float64)
    nearest, nvis = [], 0
    try:
        for pi, x in enumerate(parts):
            labs, bests, r0 = [], [], 0
            for ci, (lab, lb) in enumerate(zip(st["labels"][pi], st["lb"][pi])):
                m = lab.shape[0]
                lab2, best2, ub2, lb2, vis = kern(inputs=[x, Cm, drift, lab, lb, _u32(r0), _u32(r0 + m), pads],
                                                  template=[("D", d), ("K", k)], grid=(m * 32, 1, 1), threadgroup=(256, 1, 1),
                                                  output_shapes=[(m,), (m,), (m,), (m * k,), (m,)],
                                                  output_dtypes=[mx.uint32, mx.float32, mx.float32, mx.float32, mx.uint32])
                mx.eval(lab2, best2, ub2, lb2)
                nvis += int(mx.sum(vis))
                st["labels"][pi][ci], st["ub"][pi][ci], st["lb"][pi][ci] = lab2, ub2, lb2   # in place: one chunk over, not 2x
                labs.append(lab2); bests.append(best2)
                r0 += m
            labels = labs[0] if len(labs) == 1 else mx.concatenate(labs)
            best = bests[0] if len(bests) == 1 else mx.concatenate(bests)     # squared, as the accumulators take it
            tot += _accumulate(x, labels, best, k, accumulate)
            nearest.append((labels, best))
    except Exception:
        _bounds_state.clear()                           # chunks moved to C and chunks still at the old centres cannot mix
        raise
    st["C"] = np.array(C, dtype=np.float32, copy=True)
    st["life"] = st.get("life", 0) + 1
    st["visited"] = nvis / (sum(p.shape[0] for p in parts) * k)
    return tot, nearest


def _gpu_pass(parts, C, method="auto", accumulate="auto", iterate=False):
    """One assignment pass -> (float64 totals (k, d+2), [(labels, best) per slice] as MLX arrays)."""
    mx = _mx()
    k, d = C.shape
    for p in parts:
        if p.ndim != 2:
            raise ValueError(f"each part must be 2-D (rows, dims), got {p.ndim}-D")
    bad = [p.shape[1] for p in parts if p.shape[1] != d]
    if bad:
        raise ValueError(f"data has {bad[0]} columns but the centers have {d}; fit and predict need the same features")
    big = [p.shape for p in parts if p.size > PART_MAX_ELEMENTS]
    if big:
        raise ValueError(f"a slice of shape {big[0]} has more than {PART_MAX_ELEMENTS:,} elements, and the kernels "
                         f"address one slice with 32-bit offsets; pass the data as a list of slices of at most "
                         f"slice_rows(dims) = {slice_rows(d):,} rows (KMeans does this itself)")
    if method == "tiles" and (k % 8 or d % 8):
        raise ValueError(f"method='tiles' needs k and dims to be multiples of 8 (got k={k}, dims={d}); 'auto' picks it "
                         f"only where it applies")
    if method == "approx":
        # Two crossovers, because the fp16 multiply is ~1.4x faster than fp32 and moves where the
        # approximate path starts to pay: 64 dims if the data admits fp16, 256 if it does not.
        st = _approx_prepare(parts) if d >= APPROX_MIN_DIMS_FP16 else None
        use_half = st is not None and _approx_fp16_ok(st, C)
        if d < (APPROX_MIN_DIMS_FP16 if use_half else APPROX_MIN_DIMS):
            method = "auto"       # below the crossover the exact kernels are faster as well as exact
    if method == "approx" and accumulate == "auto":
        if use_half and st["half"] is None:
            st["half"] = [p.astype(mx.float16) for p in parts]
            mx.eval(st["half"])
        Cm = mx.array(C)
        csq = (Cm * Cm).sum(1)
        Ct = mx.contiguous(Cm.T)
        if use_half:
            Ct = Ct.astype(mx.float16)
        tot = np.zeros((k, d + 2), dtype=np.float64)
        nearest = []
        for j, x in enumerate(parts):
            xin = st["half"][j] if use_half else x
            lab = _approx_labels(xin, Ct, csq, x.shape[0], k)
            part, best = _accumulate_sorted_dist(xin if APPROX_FP16_ACCUMULATE else x, lab, k, Cm)
            tot += part
            nearest.append((lab, best))
        return tot, nearest
    casc = None
    record_bounds, cool, build = False, 0, False
    if method == "auto" and _bounds_ok(parts, k, d):
        bk = (tuple(p.shape for p in parts), k, d)
        st = _bounds_state.get(bk)
        if st is not None and not _holds(st, parts):
            st = None
        if st is not None and st["lb"] is not None:
            if _bounds_count(st, parts, C, k, d) <= BOUNDS_MAX_VISITED:
                return _bounds_pass(st, parts, C, k, d, accumulate)
            st["lb"] = None                          # decayed past use: drop the memory, keep labels/ub/C
            # Bounds that lasted one step or none were not worth building (isotropic data: the margins are
            # smaller than the drift). Wait longer before asking again, doubling each time it happens.
            st["cool"] = min(2 * st.get("cool", 0) + 1, 8) if st.get("life", 0) <= 1 else 0
        if iterate:                                  # a single pass (predict) records nothing
            if st is not None:
                cool = st.get("cool", 0)
                # Would bounds built by this pass pay at the next step? Asked of the previous step's
                # centres and drift, which is conservative: drift shrinks as the run converges.
                build = cool == 0 and _bounds_predict(st, parts, C, k, d) <= BOUNDS_MAX_VISITED
            _bounds_state.clear()                    # one dataset at a time
            record_bounds = True
            if build:
                method = "tiles"                     # the pass whose dot chunks yield the bounds
    forced = method == "cascade"                     # by name: used by the accuracy suite
    if (method == "auto" or forced) and _cascade_ok(d, k):
        # The cascade needs two things the caller does not supply: which dimensions carry the variance, and
        # where each row was last time. Both are properties of the data, so they are built once and kept
        # until a different dataset arrives. The first pass over a dataset runs normally and records its
        # labels; every pass after that can prune.
        ck = (tuple(p.shape for p in parts), k, d)
        st = _cascade_state.get(ck)
        if st is not None and not _holds(st, parts):
            st = None                                    # same shape, different data
        if st is None:
            m = _cascade_m(d)
            step = max(1, parts[0].shape[0] // 65536)
            var = _host(mx.var(parts[0][::step], axis=0))
            order = np.argsort(-var).astype(np.int64)
            _cascade_state.clear()                       # one dataset at a time; the buffers are large
            st = _hold(_cascade_state, ck, {"m": m, "order": order, "Xm": [_cascade_prefix(p, order, m) for p in parts],
                                            "labels": None, "ok": True}, parts)
        if st["labels"] is None and forced:
            Cb = mx.array(C)                         # bootstrap here so the path can be exercised directly
            base = ("tiles" if k >= TILES_MIN_K and d >= TILES_MIN_DIMS and d % 8 == 0 and k % 8 == 0
                    else "pairs" if d >= PAIRS_MIN_DIMS else "rows")
            st["labels"] = [_nearest(p, Cb, base)[0] for p in parts]
        if (st["ok"] or forced) and st["labels"] is not None:
            casc = st
    if method == "cascade":
        method = "auto"                              # if it could not apply, fall through to the usual rules
    key = None
    if method == "auto" and casc is None and _tiles1_ok(d, k):
        # Whether this path pays is a property of the data, not of the shape: its candidate window scales
        # with |x|^2, so data that was never centred - a sensor column reading around 3000, say - puts every
        # centre inside the window and every row takes the exact fallback, which costs 3x the rows kernel.
        # The answer stays exact either way, so rather than model it, the kernel reports how often it
        # happened and a dataset that goes over the limit is not offered this path again.
        seen = (tuple(id(p) for p in parts), tuple(p.shape for p in parts), k, d)
        ent = _tiles1_slow.get(seen)
        verdict = ent["slow"] if ent is not None and _holds(ent, parts) else None
        if verdict is not True:
            method = "tiles1"
            key = seen if verdict is None else None     # measure once per dataset, then trust it
    if method == "auto" and casc is None:
        # Two separate thresholds, both measured. Tiles beats rows from ~40 dims once k >= 64 (48 dims, k=256:
        # 46.4 vs 25.7 ms). Pairs beats tiles at small k from ~64 dims (64 dims, k=8: 2.7 vs 1.5 ms; 960 dims,
        # k=32: 50.4 vs 36.1 ms). Between them - 40 to 63 dims with k < 64 - rows and pairs trade places
        # unpredictably with row count, so that region stays on rows rather than chasing noise.
        if k >= TILES_MIN_K and d >= TILES_MIN_DIMS and d % 8 == 0 and k % 8 == 0:
            method = "tiles"
        elif d >= PAIRS_MIN_DIMS:
            method = "pairs"
        else:
            method = "rows"
    Cm = mx.array(C)
    tot = np.zeros((k, d + 2), dtype=np.float64)
    nearest, built = [], []
    fb = [] if key is not None else None
    plan = _fused_plan(d, k) if accumulate == "auto" and method == "rows" else None
    if casc is not None:
        worst = 0.0
        for j, x in enumerate(parts):
            labels, best, surv = _cascade_nearest(x, Cm, C, casc["Xm"][j], casc["order"],
                                                  casc["labels"][j], x.shape[0], k, d, casc["m"])
            worst = max(worst, surv)
            tot += _accumulate(x, labels, best, k, accumulate)
            nearest.append((labels, best))
        casc["labels"] = [lab for lab, _ in nearest]
        if worst > CASCADE_MAX_SURVIVORS:
            casc["ok"] = False        # this data has no usable variance structure; stop offering the path
        return tot, nearest
    for x in parts:
        if plan is not None:
            try:
                part, labels, best = _fused_pass(x, Cm, k, *plan)
                tot += part
                nearest.append((labels, best))
                continue
            except Exception:                 # a GPU that will not take the threadgroup allocation
                _fused_broken.add((d, k))
                plan = None
        lbs = [] if build else None
        labels, best = _nearest(x, Cm, method, fb, lbs)
        tot += _accumulate(x, labels, best, k, accumulate)
        nearest.append((labels, best))
        if build:
            built.append(lbs)
    if record_bounds:
        _bounds_record(parts, C, nearest, k, d, built if build else None)["cool"] = max(0, cool - 1)
    for ck, st in _cascade_state.items():
        if st["labels"] is None and ck[1] == k and ck[2] == d:
            st["labels"] = [lab for lab, _ in nearest]     # bootstrap: the next pass can prune
    if fb:
        groups = sum(f.size for f in fb)
        rate = sum(float(mx.sum(f)) for f in fb) / groups if groups else 0.0
        if len(_tiles1_slow) > 256:
            for dead in [s for s, e in _tiles1_slow.items() if any(r() is None for r in e["refs"])]:
                del _tiles1_slow[dead]
            if len(_tiles1_slow) > 256:
                _tiles1_slow.clear()                    # the verdict is only a hint
        _hold(_tiles1_slow, key, {"slow": rate > TILES1_MAX_FALLBACK}, parts)
    return tot, nearest


def assign_mlx(parts, C, method="auto", return_labels=False, accumulate="auto"):
    """One Lloyd pass on the GPU -> (per-cluster sums, counts, inertia[, labels])."""
    tot, nearest = _gpu_pass(parts, C, method, accumulate)
    result = (tot[:, 2:], np.rint(tot[:, 0]).astype(np.int64), float(tot[:, 1].sum()))
    if return_labels:
        result += (np.concatenate([_host(lab).astype(np.int64) for lab, _ in nearest]),)
    return result


def lloyd_step(parts, C, method="auto", accumulate="auto"):
    """One exact Lloyd iteration on the GPU -> (new centers float32, inertia, number of empty clusters).

    Consecutive calls on the same data at BOUNDS_MIN_DIMS+ dims keep per-centre distance bounds between them and
    skip the centres those bounds rule out (section 5 above); the labels stay exact, and only the tie rule differs
    from a single pass. The state follows the arrays by identity, so a different dataset starts afresh.

    Empty clusters follow scikit-learn's rules (_relocate_empty_clusters_dense, _average_centers): each empty cluster
    takes one of the points farthest from their assigned centers, which leaves its old cluster; a cluster still empty
    after that (only when every point sits on its center) goes to the largest cluster's center. One deliberate
    difference: when several clusters empty at once, scikit-learn pairs them with far points in np.argpartition's
    unspecified order; here the farthest point goes to the lowest-numbered empty cluster (ties: lower row index).
    """
    tot, nearest = _gpu_pass(parts, C, method, accumulate, iterate=True)
    sums, counts, inertia = tot[:, 2:], np.rint(tot[:, 0]).astype(np.int64), float(tot[:, 1].sum())
    empty = np.flatnonzero(counts == 0)
    if len(empty):
        best = np.concatenate([_host(b) for _, b in nearest])           # float32 distance to assigned center
        if best.max() > 0:
            far = np.argpartition(best, -len(empty))[-len(empty):]
            far = far[np.lexsort((far, -best[far]))]                    # farthest first, lower row index on ties
            labels = np.concatenate([_host(lab) for lab, _ in nearest]).astype(np.int64)
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
