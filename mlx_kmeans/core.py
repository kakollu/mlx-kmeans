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
TILES1_MIN_K = 64                       # ...and the k from which it wins consistently (measured; at k=32 it
                                        # ranges from 0.55x to 2.07x across dims, which is not worth picking)
TILES1_MAX_FALLBACK = 0.10              # simdgroup fallback rate past which this path stops being worth it. A
                                        # simdgroup that falls back scores all K centres exactly, which costs
                                        # about what the rows kernel costs for its 16 rows, so break-even is
                                        # near 1/3; this leaves a wide margin.
TILES1_PROBE_ROWS = 1 << 18             # rows a new dataset is measured on before the rest of the pass
TILES1_SIMDGROUPS = 4                   # simdgroups per threadgroup there (16 rows each)
MARGIN_CANDIDATES = 24                  # per-row candidate slots before falling back to a second scan
ROWS_REG_MAX_DIMS = 96                  # above this the per-thread row array spills registers (measured)
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
LANES_MAX_KW = TG_FLOATS // 2           # two k*(dims+2) tiles - the sums and their Kahan carries - on chip
LANES_BLOCKS = 512                      # simdgroups to split the rows over. What matters is the block COUNT,
                                        # not the rows per block: the best setting was ~500 blocks at 500k,
                                        # 2M and 10M rows alike, i.e. enough to fill the GPU and no more,
                                        # since every extra block re-zeroes and writes out a k*(dims+2) tile.
LANES_MIN_ROWS = 256                    # ...but not so few rows each that the tile overhead dominates
LANES_SMALL_KW = 1536                   # k*(dims+2) up to which lanes wins at every row count measured
LANES_ROWS_PER_SLOT = 4000              # ...and above it, the rows per slot from which it wins anyway.
                                        # Two costs trade off. The sorted path pays an argsort - 0.6 ns a
                                        # row, 6 ms at 10M, and slightly superlinear - but then accumulates
                                        # in registers. lanes needs no sort but re-zeroes and writes out a
                                        # k*(dims+2) tile per block, and a bigger tile also means fewer
                                        # threadgroups resident. So a small tile wins outright, and a large
                                        # one only once there are enough rows: measured lanes ahead at
                                        # kw=1088 from 2M rows, kw=2176 from 8M, kw=3328 from 16M.
LANES_ROWS_AHEAD = 8                    # rows loaded before any is added, to overlap their loads
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
    kern = (_kernel("kmeans_rows_reg", ["X", "C", "row0"], ["labels", "best_d"], _ROWS_REG_SRC)
            if d <= ROWS_REG_MAX_DIMS else
            _kernel("kmeans_rows", ["X", "C", "row0"], ["labels", "best_d"], _ROWS_SRC))
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
        lab, bst = margin(inputs=[x, Cm, csq, dot, _u32(r0), gamma],
                          template=[("D", d), ("K", k), ("M", MARGIN_CANDIDATES)],
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
    kern = _kernel(f"kmeans_tiles1_{TILES1_KEEP}", ["X", "Ct", "C", "csq", "csqmax", "row0", "gamma"],
                   ["labels", "best_d", "fb"], _tiles1_src(TILES1_KEEP), _SCORE_HDR)
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


def _nearest(x, Cm, method, fb_out=None):
    """Nearest center and its distance for every row of one slice -> (labels uint32, best float32)."""
    mx = _mx()
    n, d = x.shape
    k = Cm.shape[0]
    if method == "tiles1":
        return _tiles1_nearest(x, Cm, n, k, d, fb_out)
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
    return (np.array(total, dtype=np.float64) + np.array(total_comp, dtype=np.float64)).reshape(k, w)


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
                     template=[("D", d), ("DPL", -(-w // 32))], grid=(ns * 32, 1, 1), threadgroup=(256, 1, 1),
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
    sums, comp = acc(inputs=[x, labels, best, _u32(n)], template=[("D", d), ("K", k), ("NB", nb)],
                     grid=(nb, 1, 1), threadgroup=(min(nb, 64), 1, 1), init_value=0,
                     output_shapes=[(nb * k * w,), (nb * k * w,)], output_dtypes=[mx.float32, mx.float32])
    red = _kernel("kmeans_reduce", ["sums", "comp", "n_blocks"], ["total", "total_comp"], _REDUCE_SRC, _KAHAN)
    total, total_comp = red(inputs=[sums, comp, _u32(nb)], template=[("KW", k * w)],
                            grid=(k * w, 1, 1), threadgroup=(64, 1, 1),
                            output_shapes=[(k * w,), (k * w,)], output_dtypes=[mx.float32, mx.float32])
    return (np.array(total, dtype=np.float64) + np.array(total_comp, dtype=np.float64)).reshape(k, w)


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
    tot = np.array(out, dtype=np.float64).reshape(nb, k, w).sum(0)
    return tot, labels, best


_tiles1_slow = {}          # datasets whose error window is too wide for the one-pass path to pay off


def _tiles1_ok(d, k):
    """Shapes where the one-pass tiles path is faster (measured; see the kernel's comment).

    Dimensions and k need not be multiples of anything - the kernel pads both, exactly.
    """
    return TILES1_DIMS[0] <= d <= TILES1_DIMS[1] and k >= TILES1_MIN_K


def _gpu_pass(parts, C, method="auto", accumulate="auto"):
    """One assignment pass -> (float64 totals (k, d+2), [(labels, best) per slice] as MLX arrays)."""
    mx = _mx()
    k, d = C.shape
    key = None
    if method == "auto" and _tiles1_ok(d, k):
        # Whether this path pays is a property of the data, not of the shape: its candidate window scales
        # with |x|^2, so data that was never centred - a sensor column reading around 3000, say - puts every
        # centre inside the window and every row takes the exact fallback, which costs 3x the rows kernel.
        # The answer stays exact either way, so rather than model it, the kernel reports how often it
        # happened and a dataset that goes over the limit is not offered this path again.
        seen = (id(parts[0]), tuple(p.shape for p in parts), k, d)
        verdict = _tiles1_slow.get(seen)
        if verdict is not True:
            method = "tiles1"
            key = seen if verdict is None else None     # measure once per dataset, then trust it
    if method == "auto":
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
    nearest = []
    fb = [] if key is not None else None
    plan = _fused_plan(d, k) if accumulate == "auto" and method == "rows" else None
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
        labels, best = _nearest(x, Cm, method, fb)
        tot += _accumulate(x, labels, best, k, accumulate)
        nearest.append((labels, best))
    if fb:
        groups = sum(f.size for f in fb)
        rate = sum(float(mx.sum(f)) for f in fb) / groups if groups else 0.0
        if len(_tiles1_slow) > 256:
            _tiles1_slow.clear()                        # ids get reused; the verdict is only a hint
        _tiles1_slow[key] = rate > TILES1_MAX_FALLBACK
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
