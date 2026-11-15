"""e-means: mini-batch k-means with exponential moving averages, an opt-in GPU option beside exact Lloyd.

Method from arXiv:2611.09501 (Meta FAIR; https://e-means.github.io/), implemented from the paper's text only.
The method, in our words, and every choice we made where the paper is silent are in docs/e-means.md. In short:
per cluster j keep a count N_j and a sum S_j; the centre is S_j / N_j. Each epoch shuffles the rows and cuts
them into batches of B; each batch is assigned to the current centres, and every cluster's state moves toward
the batch's count and sum by a step a on a cosine schedule from a0 to 0.1 a0. Every 5 batches clusters whose N
fell under a tenth of the mean are revived by splitting a donor. The output is the exact mean of each cluster
over the last epoch's assignments. B and a0 come from the paper's fitted scaling laws.

e-means is APPROXIMATE: its centres are not a fixed point of Lloyd's iteration, and it is not the default.
Exact Lloyd (KMeans(algorithm="lloyd"), lloyd_step) stays the default and the reference. Within its own
definition the GPU path is exact where it can be: every batch label is the float32-nearest centre (the repo's
exact kernels), batch counts are exact, batch sums are compensated, and a rerun with the same seed gives the
same bits. Nothing here has been timed yet.

Random numbers, shared by the GPU fit and the float64 NumPy reference so both see the same draws:
    init rows      np.random.default_rng([seed, 1]).choice(n, k, replace=False)       (init="random")
                   np.random.default_rng([seed, 1]) is also the generator handed to kmeans_pp_init
    epoch e        np.random.default_rng([seed, 2, e]).permutation(n), the first floor(n/B)*B entries cut
                   into batches of B (the remainder is dropped), each batch's indices sorted ascending. Built
                   as shuffle(arange(n, uint32)), which gives the same values as permutation(n) (the shuffle's
                   draws do not depend on the dtype; tests/emeans.py --numpy checks it) at half the memory
    revival at t   np.random.default_rng([seed, 3, t]).random(k): one uniform per cluster, drawn whether or
                   not any cluster is dead, so a draw never depends on how many earlier rounds revived
Each stream is keyed on (seed, purpose, counter) rather than read sequentially from one generator: if float32
and float64 counts ever put a different number of clusters under the threshold, later draws still agree.

Laws and the reference use NumPy only; MLX is imported inside the GPU functions.
"""
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import core

# Batch law (paper Eq. 7, Table 2 "knee at 0.2%"): B = c k^bk n^bn T^bt, divided by 8 above 2M rows.
BATCH_C, BATCH_K, BATCH_N, BATCH_T = 0.035, 0.13, 0.88, 0.81
BATCH_BIG_N, BATCH_BIG_DIV = 2_000_000, 8
# Step law (paper Eq. 6, Table 2): a0 = min(c k^ak B^ab n^an T^at, 1).
STEP_C, STEP_K, STEP_B, STEP_N, STEP_T = 0.824, -0.46, 0.90, -0.45, -0.11
COSINE_FLOOR = 0.1          # a(t) anneals from a0 to this fraction of a0 (paper Eq. 5, eps)
REVIVE_EVERY = 5            # revival check after every 5th batch
REVIVE_TAU = 0.1            # dead: N_j < tau * mean(N)
SPLIT_DELTA = 2.0 ** -7     # bfloat16 machine epsilon, the paper's stated convention for delta (choice 5)
INIT_COUNT = 1e-10          # N_j at initialisation; S_j = N_j * c_j
FP16_MAX = 65504.0          # largest finite float16


# ---------------------------------------------------------------- scaling laws and schedule

def _positive_int(name, v):
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
        raise ValueError(f"{name} must be a positive integer, got {v!r}")
    return int(v)


def batch_size(n, k, T):
    """Batch size from the paper's batch law (choice 3).

    B_raw = 0.035 k^0.13 n^0.88 T^0.81, divided by 8 when n > 2,000,000 (before rounding); rounded to a power of
    two in log2, B = 2^floor(log2(B_raw) + 0.5) (half up; Python's round() is banker's at .5); then min(B, n).
    """
    n, k, T = _positive_int("n", n), _positive_int("k", k), _positive_int("T", T)
    raw = BATCH_C * k ** BATCH_K * n ** BATCH_N * T ** BATCH_T
    if n > BATCH_BIG_N:
        raw /= BATCH_BIG_DIV
    e = max(0, int(math.floor(math.log2(raw) + 0.5)))
    return int(min(2 ** e, n))


def step0(n, k, T, B):
    """Peak step a0 from the paper's decay law at batch size B (the rounded one): min(0.824 k^-0.46 B^0.90 n^-0.45 T^-0.11, 1)."""
    n, k, T, B = (_positive_int(nm, v) for nm, v in (("n", n), ("k", k), ("T", T), ("B", B)))
    return float(min(STEP_C * k ** STEP_K * B ** STEP_B * n ** STEP_N * T ** STEP_T, 1.0))


def schedule(t, M, a0):
    """Step for batch t of M, counted from 0 across epochs (choice 1): a0 (0.1 + 0.9 * 0.5 (1 + cos(pi t / M)))."""
    return a0 * (COSINE_FLOOR + (1.0 - COSINE_FLOOR) * 0.5 * (1.0 + math.cos(math.pi * t / M)))


def plan(n, k, epochs, batch_size_=None, step0_=None):
    """-> dict(B, a0, per_epoch, M): what a fit with these inputs will do. Explicit values override the laws."""
    n, k, epochs = _positive_int("n", n), _positive_int("k", k), _positive_int("epochs", epochs)
    if k > n:
        raise ValueError(f"k = {k} cannot exceed the number of rows ({n})")
    if batch_size_ is None:
        B = batch_size(n, k, epochs)
    else:
        B = _positive_int("batch_size", batch_size_)
        if B > n:
            raise ValueError(f"batch_size = {B} is larger than the number of rows ({n})")
    if step0_ is None:
        a0 = step0(n, k, epochs, B)
    else:
        a0 = float(step0_)
        if not (0.0 < a0 <= 1.0):
            raise ValueError(f"step0 must be in (0, 1], got {step0_!r}")
    per_epoch = n // B
    return dict(B=B, a0=a0, per_epoch=per_epoch, M=epochs * per_epoch)


# ---------------------------------------------------------------- shared random draws

def _seed(seed):
    if seed is None:
        return int(np.random.SeedSequence().entropy)        # fresh, but recorded so the run can be repeated
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a nonnegative integer or None, got {seed!r}")
    return int(seed)


def _init_rows(seed, n, k):
    """k distinct row indices, in draw order (centre j starts at row _init_rows(...)[j])."""
    return np.random.default_rng([seed, 1]).choice(n, size=k, replace=False)


def _epoch_batches(seed, e, n, B):
    """-> (n // B, B) row indices of epoch e: a seeded permutation, remainder dropped, each batch sorted ascending.

    Sorting within a batch changes nothing about which rows it holds; it makes the gather from a list of slices
    a set of contiguous runs and makes the batch independent of how the data is sliced.

    The values are default_rng([seed, 2, e]).permutation(n); it is built as a shuffle of a uint32 arange (the
    shuffle's draws do not depend on the dtype) and sorted in place, so the host holds 4n bytes, not the 20n
    that an int64 permutation, its sorted copy and a uint32 copy took together.
    """
    per = n // B
    p = np.arange(n, dtype=np.uint32 if n <= 2 ** 32 else np.int64)
    np.random.default_rng([seed, 2, e]).shuffle(p)
    out = p[:per * B].reshape(per, B)
    out.sort(axis=1)
    return out


def _split_factors(d):
    """(p, q): the revived slot gets c * p = c * [1+delta, 1-delta, ...], the donor c * q = c * [1-delta, 1+delta, ...]."""
    even = (np.arange(d) % 2) == 0
    return np.where(even, 1.0 + SPLIT_DELTA, 1.0 - SPLIT_DELTA), np.where(even, 1.0 - SPLIT_DELTA, 1.0 + SPLIT_DELTA)


def _revive_plan(N, seed, t, tau=REVIVE_TAU):
    """Which clusters to revive after batch t, and their donors -> (dead, donors), int64 arrays of equal length.

    Dead: N_j < tau * mean(N). Donors are drawn from the live clusters (not dead) without replacement with
    probability proportional to N, as Gumbel-top-m: keys log N_j + Gumbel noise, the m largest in order, which
    is the same distribution as m successive draws in proportion to N among those not yet drawn (choice 6).
    m = min(#dead, #live); if there are more dead clusters than live ones, the lowest-indexed dead ones are
    revived first. dead[i] is paired with donors[i]. Shared by the GPU fit (float32 N read back, converted to
    float64) and the reference (float64 N).
    """
    N = np.asarray(N, dtype=np.float64)
    k = len(N)
    u = np.random.default_rng([seed, 3, t]).random(k)
    thr = tau * (N.sum() / k)
    dead = np.flatnonzero(N < thr)
    live = np.flatnonzero(~(N < thr))
    m = min(len(dead), len(live))
    if m == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    with np.errstate(divide="ignore"):
        keys = np.log(N[live]) - np.log(-np.log(np.maximum(u[live], np.finfo(np.float64).tiny)))
    order = np.argsort(-keys, kind="stable")[:m]
    return dead[:m].astype(np.int64), live[order].astype(np.int64)


# ---------------------------------------------------------------- float64 NumPy reference

def _nearest64(x, C, chunk_terms=20_000_000):
    """Exact float64 nearest centre (lowest index on exact ties) and its squared distance, for every row of x."""
    csq = (C * C).sum(1)
    n, k = len(x), len(C)
    lab = np.empty(n, np.int64)
    best = np.empty(n)
    step = max(1, chunk_terms // max(1, k * x.shape[1]))
    for s in range(0, n, step):
        xs = x[s:s + step]
        approx = csq[None, :] - 2 * xs @ C.T
        lo = approx.min(1, keepdims=True)
        win = 1e-9 * ((xs * xs).sum(1, keepdims=True) + csq[None, :]) + 1e-300
        dist = np.full(approx.shape, np.inf)
        r, c = np.nonzero(approx <= lo + 2 * win)
        dist[r, c] = ((xs[r] - C[c]) ** 2).sum(1)
        lab[s:s + step] = dist.argmin(1)
        best[s:s + step] = dist[np.arange(len(xs)), lab[s:s + step]]
    return lab, best


def emeans_reference(X, k, epochs=3, seed=0, batch_size=None, step0=None, init="random", labels=None,
                     revivals=None, trace=None):
    """The same algorithm in float64 NumPy, with the same permutation, initial rows and donor draws.

    X: (n, d) array. It is rounded to float32 first and then computed on in float64, so the reference sees the
    data emeans_fit sees (which converts to float32); an init array is rounded the same way. init: "random" (the
    same rows the GPU fit draws) or a (k, d) array of starting centres (pass the GPU fit's own for
    init="k-means++", whose draw runs on the GPU). labels: optional list of per-batch label arrays; when given,
    the reference uses them instead of its own assignments ("driven" mode), which checks the EMA, revival and
    accumulation arithmetic independently of float32 near-ties. revivals: optional list of (dead, donors) pairs,
    one per revival check in order (M // 5 of them, as a GPU trace records them); when given they replace the
    reference's own decisions, so a cluster within float32 rounding of the threshold, or two donor keys within
    rounding of each other, cannot send the two runs apart. trace: optional list; one dict per batch is appended
    (rows, labels, C_before, N and C after the update and any revival, N_check = N before the revival at a check
    batch, revived pairs). Returns (centres float64 (k, d), record) with the record shaped like emeans_fit's.
    """
    X64 = np.asarray(X, dtype=np.float32).astype(np.float64)
    if X64.ndim != 2 or not all(X64.shape):
        raise ValueError(f"expected a nonempty 2-D (rows, dims) array, got shape {X64.shape}")
    n, d = X64.shape
    seed = _seed(seed)
    P = plan(n, k, epochs, batch_size, step0)
    B, a0, per, M = P["B"], P["a0"], P["per_epoch"], P["M"]
    if revivals is not None and len(revivals) != M // REVIVE_EVERY:
        raise ValueError(f"revivals must hold one (dead, donors) pair per check: {M // REVIVE_EVERY}, got {len(revivals)}")
    if isinstance(init, str):
        if init != "random":
            raise ValueError("the reference draws init='random' itself; for k-means++ pass the GPU fit's starting centres")
        C = X64[_init_rows(seed, n, k)].copy()
    else:
        C = np.asarray(init, dtype=np.float32).astype(np.float64)
        if C.shape != (k, d):
            raise ValueError(f"init centres must have shape {(k, d)}, got {C.shape}")
    N = np.full(k, INIT_COUNT)
    S = C * INIT_COUNT
    p, q = _split_factors(d)
    acc_cnt, acc_sum = np.zeros(k), np.zeros((k, d))
    last_labels = np.full(n, -1, np.int64)
    rec = dict(n=n, d=d, k=k, epochs=epochs, B=B, a0=a0, batches_per_epoch=per, M=M, dropped_per_epoch=n - per * B,
               seed=seed, revived=[], train_mse=[])
    t = check = 0
    for e in range(epochs):
        batches = _epoch_batches(seed, e, n, B)
        inertia = 0.0
        for i in range(per):
            rows = batches[i].astype(np.int64)
            xb = X64[rows]
            C_before = C.copy() if trace is not None else None
            if labels is None:
                lab, bd = _nearest64(xb, C)
            else:
                lab = np.asarray(labels[t], dtype=np.int64)
                bd = ((xb - C[lab]) ** 2).sum(1)
            cnt = np.bincount(lab, minlength=k).astype(np.float64)
            sums = np.zeros((k, d))
            np.add.at(sums, lab, xb)
            a = schedule(t, M, a0)
            N = (1.0 - a) * N + a * cnt
            S = (1.0 - a) * S + a * sums
            has = N > 0
            C = np.where(has[:, None], S / np.where(has, N, 1.0)[:, None], C)      # N = 0 keeps the centre (choice 4)
            inertia += float(bd.sum())
            if e == epochs - 1:
                acc_cnt += cnt
                acc_sum += sums
                last_labels[rows] = lab
            dead = donors = N_check = None
            if (t + 1) % REVIVE_EVERY == 0:          # after the 5th, 10th, ... batch (t counted from 0)
                N_check = N.copy() if trace is not None else None
                if revivals is None:
                    dead, donors = _revive_plan(N, seed, t)
                else:
                    dead, donors = (np.asarray(v, dtype=np.int64).reshape(-1) for v in revivals[check])
                    if len(dead) != len(donors):
                        raise ValueError(f"revivals[{check}]: {len(dead)} dead clusters but {len(donors)} donors")
                check += 1
                rec["revived"].append(len(dead))
                if len(dead):
                    cd = C[donors]
                    half = N[donors] * 0.5
                    N[dead], N[donors] = half, half
                    S[dead], S[donors] = (cd * p) * half[:, None], (cd * q) * half[:, None]
                    has = N > 0
                    C = np.where(has[:, None], S / np.where(has, N, 1.0)[:, None], C)
            if trace is not None:
                trace.append(dict(t=t, epoch=e, rows=rows, a=a, labels=lab, best=bd, C_before=C_before,
                                  counts=cnt, sums=sums, N=N.copy(), C=C.copy(), N_check=N_check, dead=dead,
                                  donors=donors))
            t += 1
        rec["train_mse"].append(inertia / (per * B))
    out = C.copy()
    used = acc_cnt > 0                               # a cluster empty in the last epoch keeps its EMA centre (choice 7)
    out[used] = acc_sum[used] / acc_cnt[used, None]
    rec["last_epoch_counts"] = acc_cnt
    rec["last_labels"] = last_labels                 # -1 for the rows the last epoch dropped
    rec["last_epoch_empty"] = int((~used).sum())
    return out, rec


# ---------------------------------------------------------------- GPU batch kernels (no host reads)
#
# Each returns MLX arrays and reads nothing back. They use the repo's exact kernels by name, so compiled kernels are
# shared with the Lloyd path; what differs from core's wrappers (_tiles_nearest, _tiles1_nearest, _nearest) is only
# that the transposed centre matrix is built on the GPU (core builds it from np.array(C), a host read of the
# centres, which e-means would pay every batch) and that a single distance chunk is not forced with mx.eval.

def _mx():
    import mlx.core as mx
    return mx


def _chunked(n, chunk, run):
    """Run run(r0, m) over [0, n) in chunks; force each chunk only when there is more than one (memory bound)."""
    mx = _mx()
    labels, best = [], []
    for r0 in range(0, n, chunk):
        m = min(chunk, n - r0)
        lab, bst = run(r0, m)
        if n > chunk:
            mx.eval(lab, bst)
        labels.append(lab)
        best.append(bst)
    return (labels[0], best[0]) if len(labels) == 1 else (mx.concatenate(labels), mx.concatenate(best))


def _batch_rows(x, Cm, n, k, d):
    return core._rows_kernel(x, Cm, 0, n, k, d)


def _batch_pairs(x, Cm, n, k, d):
    mx = _mx()
    pairs = core._kernel("kmeans_pairs", ["X", "C", "row0"], ["dist"], core._PAIRS_SRC)
    argmin = core._kernel("kmeans_argmin", ["dist"], ["labels", "best_d"], core._ARGMIN_SRC)

    def run(r0, m):
        dist = pairs(inputs=[x, Cm, core._u32(r0)], template=[("D", d), ("K", k)], grid=(k, m, 1),
                     threadgroup=(16, 16, 1), output_shapes=[(m, k)], output_dtypes=[mx.float32])[0]
        return argmin(inputs=[dist], template=[("K", k)], grid=(m, 1, 1), threadgroup=(64, 1, 1),
                      output_shapes=[(m,), (m,)], output_dtypes=[mx.uint32, mx.float32])
    return _chunked(n, max(1, core.DIST_BYTES // (4 * k)), run)


def _batch_tiles(x, Cm, n, k, d):
    """core._tiles_nearest with Ct built on the GPU (same values: a transpose is exact)."""
    mx = _mx()
    eps = float(np.finfo(np.float32).eps)
    csq = (Cm * Cm).sum(1)
    Ct = mx.contiguous(Cm.T)
    gamma = mx.array([d * eps / (1 - d * eps)], dtype=mx.float32)
    step = 16 if k % 16 == 0 else 8
    dot_k = (core._kernel("kmeans_dot_tiles16", ["X", "Ct", "row0"], ["dot"], core._DOT_TILES16_SRC) if step == 16
             else core._kernel("kmeans_dot_tiles", ["X", "Ct", "row0"], ["dot"], core._DOT_TILES_SRC))
    margin = core._kernel("kmeans_margin", ["X", "C", "csq", "dot", "row0", "gamma"], ["labels", "best_d"],
                          core._MARGIN_SRC)
    chunk = max(step, (core.DIST_BYTES // (4 * k)) // step * step)
    whole = n // step * step

    def run(r0, m):
        dot = dot_k(inputs=[x, Ct, core._u32(r0)], template=[("D", d), ("K", k)],
                    grid=((m // step) * (k // step) * 32, 1, 1), threadgroup=(core.TILE_THREADGROUP, 1, 1),
                    output_shapes=[(m * k,)], output_dtypes=[mx.float32])[0]
        return margin(inputs=[x, Cm, csq, dot, core._u32(r0), gamma],
                      template=[("D", d), ("K", k), ("M", core.MARGIN_CANDIDATES)],
                      grid=(m, 1, 1), threadgroup=(64, 1, 1),
                      output_shapes=[(m,), (m,)], output_dtypes=[mx.uint32, mx.float32])
    parts = []
    if whole:
        parts.append(_chunked(whole, chunk, run))
    if whole < n:
        parts.append(core._rows_kernel(x, Cm, whole, n - whole, k, d))
    if len(parts) == 1:
        return parts[0]
    return mx.concatenate([p[0] for p in parts]), mx.concatenate([p[1] for p in parts])


def _batch_tiles1(x, Cm, n, k, d, flags=None):
    """core._tiles1_nearest with the padded Ct built on the GPU and no probe; flags, if a list, gets the fallback counts."""
    mx = _mx()
    eps = float(np.finfo(np.float32).eps)
    dp, kp = -(-d // 8) * 8, -(-k // 16) * 16
    csq = (Cm * Cm).sum(1)
    csqmax = mx.max(csq).reshape(1)                    # over the real centres, before any padding
    Ct = mx.pad(Cm.T, [(0, dp - d), (0, kp - k)])     # zeros, as core's np.zeros((dp, kp)) with C.T in the corner
    if kp > k:
        csq = mx.concatenate([csq, mx.full((kp - k,), float("inf"), dtype=mx.float32)])
    gamma = mx.array([dp * eps / (1 - dp * eps)], dtype=mx.float32)
    reload = d > core.TILES1_DIMS[1]
    kern = core._kernel(f"kmeans_tiles1_{core.TILES1_KEEP}_{int(reload)}",
                        ["X", "Ct", "C", "csq", "csqmax", "row0", "gamma"], ["labels", "best_d", "fb"],
                        core._tiles1_reload_src(core.TILES1_KEEP) if reload else core._tiles1_src(core.TILES1_KEEP),
                        core._SCORE_HDR)
    whole = n // 16 * 16
    labels, best = [], []
    if whole:
        lab, bst, fell = kern(inputs=[x, Ct, Cm, csq, csqmax, core._u32(0), gamma],
                              template=[("D", d), ("DP", dp), ("K", k), ("KP", kp), ("SGPG", core.TILES1_SIMDGROUPS)],
                              grid=((whole // 16) * 32, 1, 1), threadgroup=(core.TILES1_SIMDGROUPS * 32, 1, 1),
                              output_shapes=[(whole,), (whole,), (whole // 16,)],
                              output_dtypes=[mx.uint32, mx.float32, mx.uint32])
        labels.append(lab)
        best.append(bst)
        if flags is not None:
            flags.append(fell)
    if whole < n:
        lab, bst = core._rows_kernel(x, Cm, whole, n - whole, k, d)
        labels.append(lab)
        best.append(bst)
    return (labels[0], best[0]) if len(labels) == 1 else (mx.concatenate(labels), mx.concatenate(best))


def _batch_approx(xin, x32, Cm, Ct, csq, n, k, d):
    """Labels from the matmul (xin float16 or float32), taken on trust as exact=False does; best exact from float32 rows."""
    mx = _mx()
    amin = core._kernel("kmeans_approx_argmin", ["dot", "csq"], ["labels"], core._APPROX_ARGMIN_SRC)
    per = max(1, core.DIST_BYTES // (xin.dtype.size * k))
    labels = []
    for r0 in range(0, n, per):
        m = min(per, n - r0)
        lab = amin(inputs=[xin[r0:r0 + m] @ Ct, csq], template=[("K", k)], grid=(m * 32, 1, 1),
                   threadgroup=(256, 1, 1), output_shapes=[(m,)], output_dtypes=[mx.uint32])[0]
        if n > per:
            mx.eval(lab)                      # one chunk of the product alive at a time
        labels.append(lab)
    lab = labels[0] if len(labels) == 1 else mx.concatenate(labels)
    kern = core._kernel("kmeans_label_dist", ["X", "C", "labels", "row0"], ["best_d"], core._LABEL_DIST_SRC)
    best = kern(inputs=[x32, Cm, lab, core._u32(0)], template=[("D", d)], grid=(n, 1, 1), threadgroup=(64, 1, 1),
                output_shapes=[(n,)], output_dtypes=[mx.float32])[0]
    return lab, best


def _segment_layout(labels, n, k, seg_rows=None):
    """Rows grouped by label and cut into segments of at most seg_rows rows of one cluster, planned on the GPU.

    core._accumulate_sorted plans this layout in NumPy from counts it reads back; here every step is an MLX op,
    so nothing crosses to the host. The number of segments is data-dependent, so the layout is sized for the
    most there can be: sum_j ceil(c_j / seg_rows) <= ceil(n / seg_rows) + (nonempty clusters), and there are at
    most min(k, n) of those. Segments past the real count are empty (start = end = 0) and the reduce never reads
    them. Integer scatter-add, cumsum and cummax are exact, so the layout is deterministic.
    -> (perm, seg_start, seg_end, seg_first, ns) with seg_first of length k + 1.
    """
    mx = _mx()
    s_rows = seg_rows or core.SEGMENT_ROWS
    i32 = mx.int32
    perm = mx.argsort(labels).astype(mx.uint32)
    counts = mx.zeros((k,), dtype=i32).at[labels].add(mx.array(1, dtype=i32))
    off = mx.cumsum(counts) - counts                           # first position of each cluster in perm
    nseg = (counts + (s_rows - 1)) // s_rows
    sfx = mx.cumsum(nseg) - nseg                               # first segment of each cluster
    ns = -(-n // s_rows) + min(k, n)
    total = sfx[k - 1] + nseg[k - 1]
    live = nseg > 0
    pos = mx.where(live, sfx, ns)                              # empty clusters write 0 to a sink slot
    # Each nonempty cluster writes its index at its own first segment (distinct slots), so a scatter-add
    # places it exactly; the running max then gives every segment its cluster.
    owner = mx.zeros((ns + 1,), dtype=i32).at[pos].add(mx.where(live, mx.arange(k, dtype=i32), 0))[:ns]
    owner = mx.cummax(owner)                                   # cluster of each segment
    s = mx.arange(ns, dtype=i32)
    start = off[owner] + (s - sfx[owner]) * s_rows
    end = mx.minimum(start + s_rows, off[owner] + counts[owner])
    valid = s < total
    start = mx.where(valid, start, 0).astype(mx.uint32)
    end = mx.where(valid, end, 0).astype(mx.uint32)
    seg_first = mx.concatenate([sfx, total.reshape(1)]).astype(mx.uint32)
    return perm, start, end, seg_first, ns


def _segments_totals(x, labels, best, k):
    """Sorted-segment accumulation (core's kernels) on a GPU-planned layout -> (total, total_comp), MLX, no host read."""
    mx = _mx()
    n, d = x.shape
    w = d + 2
    perm, start, end, seg_first, ns = _segment_layout(labels, n, k)
    seg = core._kernel("kmeans_segments", ["X", "best_d", "perm", "seg_start", "seg_end"], ["sums", "comp"],
                       core._SEGMENT_SRC, core._KAHAN)
    sums, comp = seg(inputs=[x, best, perm, start, end], template=[("D", d), ("DPL", -(-w // 32))],
                     grid=(ns * 32, 1, 1), threadgroup=(256, 1, 1),
                     output_shapes=[(ns * w,), (ns * w,)], output_dtypes=[mx.float32, mx.float32])
    red = core._kernel("kmeans_segment_reduce", ["sums", "comp", "seg_first"], ["total", "total_comp"],
                       core._SEGMENT_REDUCE_SRC, core._KAHAN)
    return red(inputs=[sums, comp, seg_first], template=[("W", w)], grid=(w, k, 1), threadgroup=(32, 8, 1),
               output_shapes=[(k * w,), (k * w,)], output_dtypes=[mx.float32, mx.float32])


_ACCUMULATORS = {"lanes": "_lanes_totals", "blocks": "_blocks_totals"}


def _accumulator(name):
    if name == "segments":
        return _segments_totals
    return getattr(core, _ACCUMULATORS[name])


def choose_method(B, d, k, accumulate="auto"):
    """Exact batch assignment for this shape, by the rules _gpu_pass uses minus its per-dataset paths.

    The cascade and the per-centre bounds need state about the whole dataset across passes, which batches of
    different rows do not have, so they are not offered. tiles1's probe runs once, on the first batch.
    """
    if accumulate == "auto" and core._onehot_plan(B, d, k):
        return "onehot"
    if core._tiles1_ok(d, k):
        return "tiles1"
    if k >= core.TILES_MIN_K and d >= core.TILES_MIN_DIMS and d % 8 == 0 and k % 8 == 0:
        return "tiles"
    return "pairs" if d >= core.PAIRS_MIN_DIMS else "rows"


def choose_accumulator(B, d, k):
    """lanes where its on-chip tile fits and wins (core._accumulate's rule), GPU-planned segments otherwise."""
    w = d + 2
    fits = 2 * k * w <= core.TG_FLOATS
    small = k * w <= core.LANES_SMALL_KW or B >= core.LANES_ROWS_PER_SLOT * k * w
    return "lanes" if fits and small else "segments"


def _two_sum_add(s, c, v, vc):
    """(s, c) += (v, vc) with Knuth's two-sum, elementwise in MLX: s + c carries the running total to ~float64."""
    t = s + v
    bp = t - s
    err = (s - (t - bp)) + (v - bp)
    return t, c + (err + vc)


def _gather(parts, offsets, rows):
    """Rows (sorted global indices, numpy) from a list of slices -> one (len(rows), d) MLX array, in that order."""
    mx = _mx()
    if len(parts) == 1:
        return mx.take(parts[0], mx.array(rows.astype(np.uint32)), axis=0)
    cuts = np.searchsorted(rows, offsets)
    pieces = []
    for j, p in enumerate(parts):
        a, b = cuts[j], cuts[j + 1]
        if b > a:
            pieces.append(mx.take(p, mx.array((rows[a:b].astype(np.int64) - offsets[j]).astype(np.uint32)), axis=0))
    return pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=0)


def _as_parts(X):
    mx = _mx()
    if isinstance(X, (list, tuple)):
        parts = list(X)
    elif type(X).__module__.startswith("mlx"):
        parts = [X]
    else:
        X = np.ascontiguousarray(X, dtype=np.float32)
        per = core.slice_rows(X.shape[1]) if X.ndim == 2 else max(1, len(X))
        parts = [mx.array(X[s:s + per]) for s in range(0, len(X), per)]
    if not parts:
        raise ValueError("no data")
    d = parts[0].shape[1] if parts[0].ndim == 2 else None
    for p in parts:                                     # _gpu_pass's input checks; e-means does not go through it
        if p.ndim != 2:
            raise ValueError(f"each part must be 2-D (rows, dims), got {p.ndim}-D")
        if p.shape[1] != d:
            raise ValueError(f"all parts must have the same number of columns ({d} vs {p.shape[1]})")
        if p.size > core.PART_MAX_ELEMENTS:
            raise ValueError(f"a slice of shape {p.shape} has more than {core.PART_MAX_ELEMENTS:,} elements, and the "
                             f"kernels address one slice with 32-bit offsets; pass slices of at most "
                             f"slice_rows(dims) = {core.slice_rows(d):,} rows")
        if p.dtype != mx.float32:
            raise ValueError(f"parts must be float32, got {p.dtype}")
    return parts


# ---------------------------------------------------------------- the GPU fit

def emeans_fit(parts, k, epochs=3, seed=0, batch_size=None, step0=None, init="random", assign="exact",
               method="auto", accumulate="auto", trace=None, verbose=False, _step=None):
    """e-means on the GPU -> (centres float32 (k, d), record).

    parts: a list of MLX float32 slices (as KMeans uses), one MLX array, or a NumPy array.
    epochs: T, an integer (choice 9). batch_size / step0: override the laws (B, a0); the law for a0 then uses
    the given B. init: "random" (k distinct rows, the paper's choice), "k-means++" (core.kmeans_pp_init on its
    subsample), or a (k, d) array. assign: "exact" (the repo's exact kernels) or "approx" (the exact=False
    matmul, float16 where the data's norms allow it, prepared once per fit on the full data; below that path's
    dimension thresholds the flag does nothing, as for Lloyd). method: "auto" or one of rows, pairs, tiles,
    tiles1, onehot (choose_method); accumulate: "auto" or lanes, blocks, segments (choose_accumulator).
    trace: test hook; a list that gets one dict of host copies per batch (forces a sync every batch).

    The record: B, a0, M, batches_per_epoch, dropped_per_epoch, seed, method, accumulate, assign, approx_float16,
    init, revived (clusters revived at each check), train_mse (per epoch: mean over the epoch's batch rows of the
    float32 distance to the centre each row was assigned to, as it was assigned), last_epoch_empty, timings
    (init_s, train_s, finish_s, total_s; permutation_first_wait_s, the part of epoch 0's host permutation that
    the initialisation did not cover; permutation_wait_s, the waits for epochs 1 on, each built during the
    previous epoch).
    """
    t_start = time.perf_counter()
    parts = _as_parts(parts)
    n = sum(p.shape[0] for p in parts)
    d = parts[0].shape[1]
    k = _positive_int("k", k)
    seed = _seed(seed)
    P = plan(n, k, epochs, batch_size, step0)
    B, a0, per, M = P["B"], P["a0"], P["per_epoch"], P["M"]
    w = d + 2
    if assign not in ("exact", "approx"):
        raise ValueError(f"assign must be 'exact' or 'approx', got {assign!r}")
    if accumulate not in ("auto", "lanes", "blocks", "segments"):
        raise ValueError(f"accumulate must be auto, lanes, blocks or segments, got {accumulate!r}")
    if accumulate == "lanes" and 2 * k * w > core.TG_FLOATS:
        raise ValueError(f"accumulate='lanes' needs 2 k (dims + 2) <= {core.TG_FLOATS} (got k={k}, dims={d})")
    if method not in ("auto", "rows", "pairs", "tiles", "tiles1", "onehot"):
        raise ValueError(f"method must be auto, rows, pairs, tiles, tiles1 or onehot, got {method!r}")
    if method == "tiles" and (k % 8 or d % 8):
        raise ValueError(f"method='tiles' needs k and dims to be multiples of 8 (got k={k}, dims={d})")
    if method == "tiles1" and d > core.TILES1_DIMS[1] and d % 8:
        raise ValueError(f"method='tiles1' above {core.TILES1_DIMS[1]} dims needs dims a multiple of 8 (got {d})")
    if method == "onehot" and (accumulate != "auto" or not core._onehot_plan(B, d, k)):
        raise ValueError(f"method='onehot' does not apply at k={k}, dims={d}, batch {B} (or accumulate is named)")
    if not isinstance(init, str):
        C0 = np.array(init, dtype=np.float32)
        if C0.shape != (k, d):
            raise ValueError(f"init centres must have shape {(k, d)}, got {C0.shape}")
    elif init not in ("random", "k-means++"):
        raise ValueError(f"init must be 'random', 'k-means++' or a (k, d) array, got {init!r}")

    # The next epoch's permutation is built in a worker thread while the GPU runs. Epoch 0's starts here, so it
    # overlaps the initialisation and the approximate path's preparation instead of waiting in front of batch 0.
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return _emeans_run(pool, parts, n, d, k, epochs, seed, B, a0, per, M, init, assign, method, accumulate,
                           trace, verbose, _step, t_start)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _emeans_run(pool, parts, n, d, k, epochs, seed, B, a0, per, M, init, assign, method, accumulate, trace, verbose,
                _step, t_start):
    """emeans_fit after its argument checks; pool runs the permutations."""
    mx = _mx()
    w = d + 2
    offsets = np.concatenate([[0], np.cumsum([p.shape[0] for p in parts])]).astype(np.int64)
    nxt = pool.submit(_epoch_batches, seed, 0, n, B)

    # --- initial centres
    if isinstance(init, str):
        if init == "random":
            C0 = core.take_rows(parts, _init_rows(seed, n, k))
        else:
            C0 = core.kmeans_pp_init(parts, n, k, np.random.default_rng([seed, 1]))
    else:
        C0 = np.array(init, dtype=np.float32)
    C0 = np.ascontiguousarray(C0, dtype=np.float32)
    t_init = time.perf_counter()

    # --- the batch step, chosen once per fit
    use_approx, half = False, None
    if _step is None and assign == "approx":
        st = core._approx_prepare(parts) if d >= core.APPROX_MIN_DIMS_FP16 else None
        # Decided once per fit from a bound, instead of re-read every batch. Between revivals a centre is a convex
        # combination of its previous value and batch means, so its norm never exceeds R = max(max|x|, max|c0|);
        # a split scales a centre by at most 1 + delta, and one centre is split at most once per check, so after
        # all M // 5 checks |c| <= R (1 + delta)^(M // 5). Two conditions: the repo's guard |x||c| < 65504/4
        # (its 4x margin) with one split, and, for any chain of splits however long, every converted centre
        # coordinate and every product |x.c| inside float16's range (65504). The second binds only past about
        # 178 checks, where (1 + delta)^checks passes 4.
        if st is not None:
            c0sq = float(np.einsum("ij,ij->i", C0, C0).max())
            if np.isfinite(st["xsqmax"]) and np.isfinite(c0sq):
                xr, R = math.sqrt(st["xsqmax"]), math.sqrt(max(st["xsqmax"], c0sq))
                growth = math.exp(min(700.0, (M // REVIVE_EVERY) * math.log1p(SPLIT_DELTA)))
                use_half = (xr * R * (1.0 + SPLIT_DELTA) < core.APPROX_FP16_LIMIT
                            and R * growth < FP16_MAX and xr * R * growth < FP16_MAX)
            else:
                use_half = False
        else:
            use_half = False
        use_approx = d >= (core.APPROX_MIN_DIMS_FP16 if use_half else core.APPROX_MIN_DIMS)
        if use_approx and use_half:
            if st["half"] is None:                         # the dataset's float16 copy, shared with Lloyd's exact=False
                st["half"] = [p.astype(mx.float16) for p in parts]
                mx.eval(st["half"])
            half = st["half"]
    if _step is not None:
        meth, acc_name = "test-hook", "test-hook"
    else:
        meth = choose_method(B, d, k, accumulate) if method == "auto" else method
        if use_approx:
            meth = "approx"
        acc_name = "fused" if meth == "onehot" else (choose_accumulator(B, d, k) if accumulate == "auto" else accumulate)
    accumulate_fn = None if acc_name in ("fused", "test-hook") else _accumulator(acc_name)
    onehot_plan = core._onehot_plan(B, d, k) if meth == "onehot" else None

    def step(xb, rows, C, probe):
        """-> labels, best, total, comp for one batch against centres C (MLX)."""
        if _step is not None:
            return _step(xb, C)
        if meth == "onehot":
            total, comp, labels, best = core._onehot_totals(xb, C, k, onehot_plan)
            return labels, best, total, comp
        if meth == "approx":
            Ct = mx.contiguous(C.T)
            xin = xb
            if half is not None:
                Ct = Ct.astype(mx.float16)
                xin = _gather(half, offsets, rows)
                # The norm from the same converted centre as the product (paper App J.1, Eqs. 9-11), not from
                # the float32 one as Lloyd's exact=False does. The float16 product's own output rounding stays.
                csq = mx.sum(mx.square(Ct.astype(mx.float32)), axis=0)
            else:
                csq = (C * C).sum(1)
            labels, best = _batch_approx(xin, xb, C, Ct, csq, B, k, d)
        elif meth == "tiles1":
            labels, best = _batch_tiles1(xb, C, B, k, d, probe)
        elif meth == "tiles":
            labels, best = _batch_tiles(xb, C, B, k, d)
        elif meth == "pairs":
            labels, best = _batch_pairs(xb, C, B, k, d)
        else:
            labels, best = _batch_rows(xb, C, B, k, d)
        total, comp = accumulate_fn(xb, labels, best, k)
        return labels, best, total, comp

    # --- state on the GPU
    p_np, q_np = _split_factors(d)
    p_mx, q_mx = mx.array(p_np.astype(np.float32)), mx.array(q_np.astype(np.float32))
    C = mx.array(C0)
    N = mx.full((k,), INIT_COUNT, dtype=mx.float32)
    S = C * INIT_COUNT
    acc_s = mx.zeros((k * w,), dtype=mx.float32)
    acc_c = mx.zeros((k * w,), dtype=mx.float32)
    batch_inertia = []
    revived = []
    perm_first_wait = perm_wait = 0.0
    probe_rate = None
    t = 0
    for e in range(epochs):
        tw = time.perf_counter()
        batches = nxt.result()
        if e == 0:
            perm_first_wait = time.perf_counter() - tw       # what init did not cover of epoch 0's permutation
        else:
            perm_wait += time.perf_counter() - tw            # steady state: overlapped with the previous epoch
        if e + 1 < epochs:
            nxt = pool.submit(_epoch_batches, seed, e + 1, n, B)
        last = e == epochs - 1
        for i in range(per):
            rows = batches[i]
            xb = _gather(parts, offsets, rows)
            probe = [] if (t == 0 and meth == "tiles1" and method == "auto") else None
            C_before = C
            if t == 0 and meth == "onehot":
                try:                                     # a GPU that will not take the kernel: rows + an accumulator
                    labels, best, total, comp = step(xb, rows, C, probe)
                    mx.eval(labels, best, total, comp)
                except Exception:
                    core._onehot_broken.add((d, k))
                    meth, acc_name = "rows", choose_accumulator(B, d, k)
                    accumulate_fn = _accumulator(acc_name)
                    labels, best, total, comp = step(xb, rows, C, None)
            else:
                labels, best, total, comp = step(xb, rows, C, probe)
            if probe:
                # tiles1 pays only when few simdgroups fall back (uncentred data puts every centre in the window);
                # measured once, on the first batch, as _gpu_pass does once per dataset.
                fell = probe[0]
                rate = float(mx.sum(fell).item()) / max(1, fell.size)
                probe_rate = rate
                if rate > core.TILES1_MAX_FALLBACK:
                    meth = ("tiles" if k >= core.TILES_MIN_K and d >= core.TILES_MIN_DIMS and d % 8 == 0
                            and k % 8 == 0 else "pairs" if d >= core.PAIRS_MIN_DIMS else "rows")
            tot = (total + comp).reshape(k, w)
            a = schedule(t, M, a0)
            N = N * (1.0 - a) + tot[:, 0] * a
            S = S * (1.0 - a) + tot[:, 2:] * a
            C = mx.where((N > 0)[:, None], S / N[:, None], C)
            batch_inertia.append(mx.sum(tot[:, 1]))
            if last:
                acc_s, acc_c = _two_sum_add(acc_s, acc_c, total, comp)
            dead = donors = Nh = None
            if (t + 1) % REVIVE_EVERY == 0:          # after the 5th, 10th, ... batch (t counted from 0)
                mx.eval(N)
                Nh = np.array(N)                     # the k counts the decision is made from
                dead, donors = _revive_plan(Nh, seed, t)
                revived.append(len(dead))
                if len(dead):
                    di = mx.array(dead.astype(np.uint32))
                    oi = mx.array(donors.astype(np.uint32))
                    cd = C[oi]                                   # live donors: N > 0, so C = S / N there
                    hn = N[oi] * 0.5
                    N[di] = hn
                    N[oi] = hn
                    S[di] = (cd * p_mx) * hn[:, None]
                    S[oi] = (cd * q_mx) * hn[:, None]
                    C = mx.where((N > 0)[:, None], S / N[:, None], C)
            if trace is not None:
                mx.eval(labels, best, total, comp, N, S, C, acc_s, acc_c, batch_inertia[-1])
                trace.append(dict(t=t, epoch=e, rows=np.asarray(rows, np.int64), a=a, method=meth,
                                  labels=np.array(labels).astype(np.int64), best=np.array(best),
                                  C_before=np.array(C_before),
                                  counts=np.array(total, np.float64).reshape(k, w)[:, 0]
                                  + np.array(comp, np.float64).reshape(k, w)[:, 0],
                                  sums=np.array(total, np.float64).reshape(k, w)[:, 2:]
                                  + np.array(comp, np.float64).reshape(k, w)[:, 2:],
                                  N=np.array(N), C=np.array(C), N_check=Nh, dead=dead, donors=donors))
            else:
                mx.async_eval(N, S, C, acc_s, acc_c, batch_inertia[-1])
            t += 1
    mx.eval(N, S, C, acc_s, acc_c, *batch_inertia)
    t_train = time.perf_counter()
    tot = (np.array(acc_s, np.float64) + np.array(acc_c, np.float64)).reshape(k, w)
    cnt = np.rint(tot[:, 0])
    out = np.array(C, dtype=np.float32)
    used = cnt > 0                                   # a cluster empty in the last epoch keeps its EMA centre (choice 7)
    out[used] = (tot[used, 2:] / cnt[used, None]).astype(np.float32)
    inert = np.array(mx.stack(batch_inertia), dtype=np.float64)
    train_mse = [float(inert[e * per:(e + 1) * per].sum() / (per * B)) for e in range(epochs)]
    t_end = time.perf_counter()
    rec = dict(n=n, d=d, k=k, epochs=epochs, B=B, a0=a0, batches_per_epoch=per, M=M, dropped_per_epoch=n - per * B,
               seed=seed, init=init if isinstance(init, str) else "given", assign="approx" if use_approx else "exact",
               approx_float16=half is not None, method=meth, accumulate=acc_name, tiles1_fallback_rate=probe_rate,
               revived=revived, train_mse=train_mse, last_epoch_empty=int((~used).sum()),
               init_centres=C0 if trace is not None else None,
               timings=dict(init_s=t_init - t_start, train_s=t_train - t_init, finish_s=t_end - t_train,
                            total_s=t_end - t_start, permutation_first_wait_s=perm_first_wait,
                            permutation_wait_s=perm_wait))
    if verbose:
        print(f"e-means: B={B} a0={a0:.4f} M={M} ({per} batches x {epochs} epochs, {n - per * B} rows dropped per "
              f"epoch, {(n - per * B) / n:.1%}), method {meth}/{acc_name}; train MSE by epoch "
              + ", ".join(f"{v:.6e}" for v in train_mse) + f"; revived {sum(revived)} in {len(revived)} checks; "
              f"waited {perm_first_wait:.3f} s for the first permutation, {perm_wait:.3f} s for the rest")
    return out, rec
