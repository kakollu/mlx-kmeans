"""Exact two-stage assignment: prune on a variance-ordered prefix, then score survivors exactly.

A partial sum over the first m dimensions lower-bounds the full distance, so a centre whose partial already
exceeds a distance we have ACHIEVED (the row's previous centre, scored exactly) is provably not the nearest.
Stage 1 runs the fused tile matmul over those m dims and writes one bit per centre. Stage 2 walks the set
bits and scores those centres exactly. Output must be identical to scoring every centre.
"""
import sys, time
import numpy as np, mlx.core as mx
sys.path.insert(0, "/Users/satya/git/kmeans")
from mlx_kmeans import core

# Stage 1: same structure as the one-pass tiles kernel, but emitting a survivor mask instead of an argmin.
MASK = """
    uint sg = thread_position_in_grid.x / 32, lane = thread_position_in_grid.x % 32;
    uint rbase = sg * 16, xrow = rbase * M;
    const uint MK = M / 8;
    threadgroup float tile[SGPG * 256];
    threadgroup float *T = tile + (thread_position_in_threadgroup.x / 32) * 256;
    simdgroup_float8x8 a0[MK], a1[MK];
    for (uint kk = 0; kk < MK; kk++) {
        simdgroup_load(a0[kk], Xm + xrow + kk * 8, M);
        simdgroup_load(a1[kk], Xm + xrow + 8 * M + kk * 8, M);
    }
    uint i = rbase + lane, xi = i * M;
    float xsq = 0;
    if (lane < 16) for (uint j = 0; j < M; j++) xsq += Xm[xi + j] * Xm[xi + j];
    float g3 = 3 * gamma[0];
    float u = (lane < 16) ? ub[i] : 0.0f;
    uint word = 0;
    for (uint ct = 0; ct < K / 16; ct++) {
        uint c0 = ct * 16;
        simdgroup_float8x8 b0, b1;
        simdgroup_float8x8 acc00 = simdgroup_float8x8(0.0f), acc01 = simdgroup_float8x8(0.0f);
        simdgroup_float8x8 acc10 = simdgroup_float8x8(0.0f), acc11 = simdgroup_float8x8(0.0f);
        for (uint kk = 0; kk < MK; kk++) {
            simdgroup_load(b0, Ct + kk * 8 * K + c0, K);
            simdgroup_load(b1, Ct + kk * 8 * K + c0 + 8, K);
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
                // partial distance over the first M dims, expanded form, then widened into a LOWER bound
                float a = xsq + csq[c] - 2 * T[lane * 16 + t];
                float lb = a - g3 * (xsq + csq[c]);
                if (!(lb > u)) bits |= (1u << t);          // keep unless provably worse than an achieved centre
            }
            word |= bits << ((ct & 1u) * 16);
            if (ct & 1u) { mask[i * (K / 32) + (ct >> 1)] = word; word = 0; }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

# Stage 2: walk the set bits, score those centres exactly. Row held once across the lanes.
SCORE = """
    uint i = thread_position_in_grid.x / 32, lane = thread_position_in_grid.x % 32;
    const uint DPL = (D + 31) / 32, W = K / 32;
    float xv[DPL];
    for (uint c = 0; c < DPL; c++) { uint j = lane + c * 32; xv[c] = (j < D) ? X[i * D + j] : 0.0f; }
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

def med(f, r=3):
    mx.eval(f()); ts = []
    for _ in range(r):
        t = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t)
    return float(np.median(ts))

def cascade(x, Xm, Cm, Ctm, csq_m, gamma, ub, n, k, d, m, want=None):
    W = k // 32
    k1 = core._kernel(f"casc_mask_{m}_{k}", ["Xm", "Ct", "csq", "ub", "gamma"], ["mask"], MASK)
    mask = k1(inputs=[Xm, Ctm, csq_m, ub, gamma],
              template=[("M", m), ("K", k), ("SGPG", 4)],
              grid=((n // 16) * 32, 1, 1), threadgroup=(4 * 32, 1, 1),
              output_shapes=[(n * W,)], output_dtypes=[mx.uint32])[0]
    k2 = core._kernel(f"casc_score_{d}_{k}", ["X", "C", "mask"], ["labels", "best_d"], SCORE)
    lab, bst = k2(inputs=[x, Cm, mask], template=[("D", d), ("K", k)],
                  grid=(n * 32, 1, 1), threadgroup=(256, 1, 1),
                  output_shapes=[(n,), (n,)], output_dtypes=[mx.uint32, mx.float32])
    if want is not None:
        want.append(mask)
    return lab, bst

rng = np.random.default_rng(0)
eps = float(np.finfo(np.float32).eps)
from bench import read_fvecs
from pathlib import Path
ROOT = Path("/Users/satya/git/kmeans")

for tag, path, k, m, iters in [("SIFT1M", "sift", 1024, 64, 5), ("SIFT1M", "sift", 1024, 96, 5),
                               ("SIFT1M", "sift", 256, 64, 5), ("SIFT1M", "sift", 4096, 64, 5)]:
    X = read_fvecs(ROOT / f"data/{path}/{path}_base.fvecs")
    n, d = X.shape
    parts = [mx.array(X)]; mx.eval(parts)
    C = core.kmeans_pp_init(parts, n, k, np.random.default_rng(0))
    for _ in range(iters - 1):
        C, _, _ = core.lloyd_step(parts, C)
    C_before = C.copy()                      # centres BEFORE the step we are about to assign for
    C, _, _ = core.lloyd_step(parts, C)      # centres moved; labels below are genuinely stale
    order = np.argsort(-X[::97].var(0))
    x, Cm = parts[0], mx.array(C)
    Xm = mx.array(np.ascontiguousarray(X[:, order[:m]]))
    Cmm = np.ascontiguousarray(C[:, order[:m]])
    Ctm = mx.array(np.ascontiguousarray(Cmm.T))
    csq_m = mx.array((Cmm * Cmm).sum(1).astype(np.float32))
    gamma = mx.array([m * eps / (1 - m * eps)], dtype=mx.float32)
    mx.eval(x, Cm, Xm, Ctm, csq_m, gamma)

    ref_l, ref_b = core._nearest(x, Cm, "tiles"); mx.eval(ref_l, ref_b)
    # the bound we really have: last iteration's label (from C_before), scored against the MOVED centres
    _, _, _, lab_prev = core.assign_mlx(parts, C_before, return_labels=True)
    stale = float((lab_prev != np.array(ref_l).astype(np.int64)).mean())
    lab_prev = mx.array(lab_prev.astype(np.uint32))
    dk = core._kernel("casc_ub", ["X", "C", "labels", "row0"], ["best_d"], core._LABEL_DIST_SRC)
    ub = dk(inputs=[x, Cm, lab_prev, core._u32(0)], template=[("D", d)], grid=(n, 1, 1),
            threadgroup=(64, 1, 1), output_shapes=[(n,)], output_dtypes=[mx.float32])[0]
    mx.eval(ub)

    got = []
    l2, b2 = cascade(x, Xm, Cm, Ctm, csq_m, gamma, ub, n, k, d, m, got)
    mx.eval(l2, b2, got[0])
    bits = np.unpackbits(np.array(got[0]).view(np.uint8))
    surv = bits.sum() / (n * k)
    bad_l = int(mx.sum(l2 != ref_l))
    bad_b = int(mx.sum(b2 != ref_b))
    rel = float(mx.max(mx.abs(b2 - ref_b) / mx.maximum(ref_b, 1e-30))) if bad_b else 0.0
    t_ref = med(lambda: mx.eval(core._nearest(x, Cm, "tiles")))
    def full_cascade():
        u = dk(inputs=[x, Cm, lab_prev, core._u32(0)], template=[("D", d)], grid=(n, 1, 1),
               threadgroup=(64, 1, 1), output_shapes=[(n,)], output_dtypes=[mx.float32])[0]
        return cascade(x, Xm, Cm, Ctm, csq_m, gamma, u, n, k, d, m)
    t_ub = med(lambda: mx.eval(dk(inputs=[x, Cm, lab_prev, core._u32(0)], template=[("D", d)],
                                  grid=(n, 1, 1), threadgroup=(64, 1, 1),
                                  output_shapes=[(n,)], output_dtypes=[mx.float32])[0]))
    t_cas = med(lambda: mx.eval(full_cascade()))
    print(f"{tag} {n//1000}k x {d} k={k}, prune on {m:>3}   today {t_ref*1e3:6.1f}  cascade {t_cas*1e3:6.1f}"
          f"  {t_ref/t_cas:5.2f}x   (bound {t_ub*1e3:4.1f} incl)   survivors {surv:6.2%}"
          f"   wrong labels {bad_l}   stale labels {stale:.1%}")
    del X, parts, x, Cm, Xm, Ctm

# ---------------------------------------------------------------------------------------------------
# Status, 2026-09-24. Working prototype, not integrated. Measured on SIFT1M with a genuinely stale
# upper bound (6.9% of labels out of date), zero wrong labels in every case:
#
#     1M x 128, k=1024, prune on 64 dims    33.5 -> 18.0 ms   1.86x   4.21% survive
#     1M x 128, k=1024, prune on 96 dims    33.5 -> 17.5 ms   1.92x   0.55% survive
#     1M x 128, k=4096, prune on 64 dims   132.6 -> 57.2 ms   2.32x   2.12% survive
#     1M x 128, k=256,  prune on 64 dims     9.5 ->  6.6 ms   1.43x   7.79% survive
#
# What is still needed to ship it:
#   - Data dependence. On isotropic data (equal variance per dimension) a half-prefix is half of EVERY
#     distance, so nothing prunes: 49% survive and it runs at 0.45x. It needs the same runtime probe the
#     one-pass path uses - measure the survivor rate on a prefix of rows, fall back if it is too high.
#   - Memory. Stage 1 reads a variance-ordered prefix, which is a separate n x m buffer: +50% of X at
#     m = d/2. Permuting X once per fit instead would avoid it but changes the layout everywhere.
#   - Bootstrap. The first iteration has no previous labels; seed the bound from kmeans_pp_init's
#     distances or run one ordinary pass.
#   - best_d ordering. Stage 2 reduces with simd_sum, so distances differ from the other paths in the
#     last bits (9.8e-07 relative). Labels are unaffected; inertia would move by ~1e-7.
#   - Reach. Stage 1 holds the prefix tiles in registers, so m <= 96 - which is why this helps at 128-256
#     dims and cannot help GIST at 960, where a useful prefix would be ~480.
