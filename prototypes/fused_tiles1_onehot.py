"""Step 3 prototype: tiles1's assignment (16 rows per simdgroup, X tiles in registers, R=4 candidates verified exactly)
with the one-hot accumulation multiplying those same tiles. Partials per simdgroup, Kahan reduce. n % 16 == 0 here."""
import sys, time, numpy as np, mlx.core as mx
sys.path.insert(0, "/Users/satya/git/kmeans"); from mlx_kmeans import core
R = 4
def med(f, r=5):
    f(); mx.synchronize(); ts = []
    for _ in range(r):
        a = time.perf_counter(); f(); mx.synchronize(); ts.append((time.perf_counter() - a) * 1e3)
    return np.median(ts)
def best3(f):
    med(f); return min(med(f) for _ in range(3))
keep = "".join(f"                bool l{j} = a < v{j};\n" for j in range(R))
for j in range(R - 1, 0, -1):
    keep += (f"                v{j} = l{j-1} ? v{j-1} : (l{j} ? a : v{j});\n"
             f"                u{j} = l{j-1} ? u{j-1} : (l{j} ? c : u{j});\n")
keep += "                v0 = l0 ? a : v0;  u0 = l0 ? c : u0;\n"
SRC = """
    uint lane = thread_position_in_grid.x % 32, sg = thread_position_in_grid.x / 32;
    uint sgl = thread_position_in_threadgroup.x / 32;
    const uint DK = DP / 8, DFULL = D / 8;
    threadgroup float tile[SGPG * 256];
    threadgroup float *T = tile + sgl * 256;
    threadgroup float ohall[SGPG * K * 8];
    threadgroup float *oh = ohall + sgl * K * 8;
    for (uint e = lane; e < K * 8; e += 32) oh[e] = 0.0f;
    simdgroup_float8x8 acc[KB][DK], accx[KB];
    #pragma clang loop unroll(full)
    for (uint cb = 0; cb < KB; cb++) {
        accx[cb] = simdgroup_float8x8(0.0f);
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < DK; kk++) acc[cb][kk] = simdgroup_float8x8(0.0f);
    }
    float g3 = 3 * gamma[0];
    uint base = sg * RPS, end = min(base + RPS, uint(N));
    float fell = 0.0f;
    for (uint rbase = base; rbase < end; rbase += 16) {
        uint xrow = rbase * D;
        simdgroup_float8x8 a0[DK], a1[DK];
        for (uint kk = 0; kk < DFULL; kk++) {
            simdgroup_load(a0[kk], X + xrow + kk * 8, D);
            simdgroup_load(a1[kk], X + xrow + 8 * D + kk * 8, D);
        }
        if (DFULL < DK) {
            for (uint e = lane; e < 128; e += 32) {
                uint r = e / 8, j = DFULL * 8 + (e % 8);
                T[e] = (j < D) ? X[xrow + r * D + j] : 0.0f;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_load(a0[DFULL], T, 8);
            simdgroup_load(a1[DFULL], T + 64, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint i = rbase + lane, xi = i * D;
        float xsq = 0;
        if (lane < 16) for (uint j = 0; j < D; j++) xsq += X[xi + j] * X[xi + j];
""" + "".join(f"        float v{j} = INFINITY; uint u{j} = 0;\n" for j in range(R)) + """
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
""" + f"        bool wide = (lane < 16) && (v{R-1} <= v0 + g3 * (xsq + csq[u0]) + g3 * (xsq + csqmax[0]));\n" + """
        fell = max(fell, simd_max(wide ? 1.0f : 0.0f));
        float best = INFINITY; uint bl = 0;
        if (lane < 16) {
            if (wide) {
                for (uint c = 0; c < K; c++) {
                    float s = 0;
                    for (uint j = 0; j < D; j++) { float q = X[xi + j] - C[c * D + j]; s += q * q; }
                    bool lt = s < best; best = select(best, s, lt); bl = select(bl, c, lt);
                }
            } else {
""" + "".join(f"                SCORE(u{j})\n" for j in range(R)) + """
            }
            labels[i] = bl; best_d[i] = best;
        }
        #pragma clang loop unroll(full)
        for (uint h = 0; h < 2; h++) {
            bool mine = (lane / 8) == h;
            uint r = lane % 8;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) T[e] = 0.0f;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (mine) { oh[bl * 8 + r] = 1.0f; T[r * 8] = 1.0f; T[r * 8 + 1] = best; }
            uint pres = simd_or(mine ? (1u << (bl / 8)) : 0u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_float8x8 bx;
            simdgroup_load(bx, T, 8);
            #pragma clang loop unroll(full)
            for (uint cb = 0; cb < KB; cb++) {
                if (SKIP && !(pres & (1u << cb))) continue;
                simdgroup_float8x8 A;
                simdgroup_load(A, oh + cb * 64, 8);
                simdgroup_multiply_accumulate(accx[cb], A, bx, accx[cb]);
                #pragma clang loop unroll(full)
                for (uint kk = 0; kk < DK; kk++) {
                    simdgroup_float8x8 ah = (h == 0) ? a0[kk] : a1[kk];
                    simdgroup_multiply_accumulate(acc[cb][kk], A, ah, acc[cb][kk]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (mine) oh[bl * 8 + r] = 0.0f;
        }
    }
    if (lane == 0) fb[sg] = (uint)fell;
    #pragma clang loop unroll(full)
    for (uint cb = 0; cb < KB; cb++) {
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < DK; kk++) simdgroup_store(acc[cb][kk], partial + (sg * K + cb * 8) * WPAD + kk * 8, WPAD);
        simdgroup_store(accx[cb], partial + (sg * K + cb * 8) * WPAD + DP, WPAD);
    }
"""
REDUCE = """
    uint col = thread_position_in_grid.x, c = thread_position_in_grid.y;
    if (col >= W) return;
    uint src = (col == 0) ? DP : (col == 1) ? DP + 1 : col - 2;
    float s = 0.0f, cc = 0.0f;
    for (uint g = 0; g < NSG; g++) KAHAN_ADD(s, cc, partial[(g * K + c) * WPAD + src]);
    total[c * W + col] = s; comp[c * W + col] = cc;
"""
def run(x, Cm, n, d, k, rps, sgpg, skip):
    dp, kp = -(-d // 8) * 8, -(-k // 16) * 16
    assert k % 8 == 0 and n % 16 == 0
    csq = (Cm * Cm).sum(1); csqmax = mx.max(csq).reshape(1)
    Ctp = np.zeros((dp, kp), np.float32); Ctp[:d, :k] = np.array(Cm).T; Ct = mx.array(Ctp)
    if kp > k: csq = mx.concatenate([csq, mx.full((kp - k,), float("inf"), dtype=mx.float32)])
    eps = float(np.finfo(np.float32).eps); gamma = mx.array([dp * eps / (1 - dp * eps)], dtype=mx.float32)
    wpad, kb, dk, nsg = dp + 8, k // 8, dp // 8, -(-n // rps)
    kern = core._kernel(f"fused_t1_{d}_{k}_{sgpg}_{int(skip)}", ["X", "Ct", "C", "csq", "csqmax", "gamma"], ["labels", "best_d", "fb", "partial"], SRC, core._SCORE_HDR)
    lab, best, fb, part = kern(inputs=[x, Ct, Cm, csq, csqmax, gamma],
                               template=[("D", d), ("DP", dp), ("K", k), ("KP", kp), ("KB", kb), ("WPAD", wpad), ("SGPG", sgpg), ("RPS", rps), ("N", n), ("SKIP", 1 if skip else 0)],
                               grid=(nsg * 32, 1, 1), threadgroup=(sgpg * 32, 1, 1),
                               output_shapes=[(n,), (n,), (nsg,), (nsg * k * wpad,)], output_dtypes=[mx.uint32, mx.float32, mx.uint32, mx.float32])
    red = core._kernel("fused_t1_reduce", ["partial"], ["total", "comp"], REDUCE, core._KAHAN)
    w = d + 2
    t, c = red(inputs=[part], template=[("D", d), ("DP", dp), ("K", k), ("W", w), ("WPAD", wpad), ("NSG", nsg)], grid=(-(-w // 32) * 32, k, 1), threadgroup=(32, 1, 1),
               output_shapes=[(k * w,), (k * w,)], output_dtypes=[mx.float32, mx.float32])
    mx.eval(lab, best, t, c, fb)
    return (core._host(t, np.float64) + core._host(c, np.float64)).reshape(k, w), lab, best, float(mx.mean(fb))
def shape(tag, n, d, k, sgpgs=(1, 2, 4), rps=1024):
    P = core.make_data_mlx(n, d, k, 0); x = P[0]; mx.eval(x)
    C = core.kmeans_pp_init(P, n, k, np.random.default_rng(0)); Cm = mx.array(C); mx.eval(Cm)
    t_auto = best3(lambda: core.lloyd_step(P, C))
    lab_ref, best_ref = core._nearest(x, Cm, "rows"); mx.eval(lab_ref, best_ref); ref = core._accumulate(x, lab_ref, best_ref, k, "sorted")
    print(f"{tag:>24} | auto {t_auto:5.2f} ms | floor {n*d*4/490e9*1e3:4.2f} | TG mem/simdgroup {(k*8+256)*4/1024:4.1f} KB")
    for sgpg in sgpgs:
        for skip in ((True, False) if k <= 64 else (False, True)):
            try:
                tot, lab, best, fell = run(x, Cm, n, d, k, rps, sgpg, skip)
            except Exception as e:
                print(f"   {sgpg}/TG skip {str(skip):5s}: failed - {str(e).splitlines()[-1][:80]}"); continue
            ok = bool(mx.all(lab == lab_ref)) and bool(mx.all(best == best_ref)); err = np.abs(tot - ref).max() / np.abs(ref).max()
            t = best3(lambda: run(x, Cm, n, d, k, rps, sgpg, skip))
            print(f"   {sgpg}/TG skip {str(skip):5s}: {t:5.2f} ms  ({t_auto / t:4.2f}x)  labels {'=' if ok else 'DIFF'}, totals {err:.0e}, wide tiles {100 * fell:.1f}%")
shape("single-cell 2M x 50 k64", 2_000_000, 50, 64)
shape("satellite 10M x 12 k32", 10_000_000, 12, 32)
shape("geo-trips 10M x 4 k256", 10_000_000, 4, 256, (1, 2))
shape("2M x 50 k32", 2_000_000, 50, 32)
shape("2M x 30 k64", 2_000_000, 30, 64)
