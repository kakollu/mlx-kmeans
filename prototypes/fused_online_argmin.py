"""Codex's recommended kernel, built as specified: fused online-argmin approximate assignment.

- 16 rows per simdgroup; CPT centre tiles (16 centres each) computed per reduction step
- row tiles re-read from cache per centre step (register cap otherwise), centre tiles streamed
- ALL 32 lanes reduce: lane l owns row l/2 and half l%2 of the centres; pairs combined by shuffle_xor
- named accumulators only (2-D arrays of simdgroup matrices spill)
- IT = float or half inputs; accumulation always float
Compared against the current approximate assignment (mx.matmul + one fused argmin pass)."""
import sys, time
import numpy as np, mlx.core as mx
sys.path.insert(0, "/Users/satya/git/kmeans")
from mlx_kmeans import core
from bench import read_fvecs
from pathlib import Path
ROOT = Path("/Users/satya/git/kmeans")

def src(cpt, it):
    W = 16 * cpt
    decl = ""
    for c in range(cpt):
        decl += (f"        simdgroup_matrix<{it},8,8> b0_{c}, b1_{c};\n"
                 + "".join(f"        simdgroup_float8x8 acc{c}_{p} = simdgroup_float8x8(0.0f);\n" for p in ("00","01","10","11")))
    loads = "".join(f"            simdgroup_load(b0_{c}, Ct + kk * 8 * K + c0 + {c*16}, K);\n"
                    f"            simdgroup_load(b1_{c}, Ct + kk * 8 * K + c0 + {c*16} + 8, K);\n" for c in range(cpt))
    macs = "".join(f"            simdgroup_multiply_accumulate(acc{c}_00, a0, b0_{c}, acc{c}_00);\n"
                   f"            simdgroup_multiply_accumulate(acc{c}_01, a0, b1_{c}, acc{c}_01);\n"
                   f"            simdgroup_multiply_accumulate(acc{c}_10, a1, b0_{c}, acc{c}_10);\n"
                   f"            simdgroup_multiply_accumulate(acc{c}_11, a1, b1_{c}, acc{c}_11);\n" for c in range(cpt))
    stores = "".join(f"        simdgroup_store(acc{c}_00, T + {c*16}, {W});\n"
                     f"        simdgroup_store(acc{c}_01, T + {c*16} + 8, {W});\n"
                     f"        simdgroup_store(acc{c}_10, T + 8 * {W} + {c*16}, {W});\n"
                     f"        simdgroup_store(acc{c}_11, T + 8 * {W} + {c*16} + 8, {W});\n" for c in range(cpt))
    return f"""
    uint sg = thread_position_in_grid.x / 32, lane = thread_position_in_grid.x % 32;
    uint rbase = sg * 16, xrow = rbase * D;
    threadgroup float tile[SGPG * {16*W}];
    threadgroup float *T = tile + (thread_position_in_threadgroup.x / 32) * {16*W};
    uint r = lane >> 1, h = lane & 1;
    float best = INFINITY; uint bl = 0;
    for (uint c0 = 0; c0 < K; c0 += {W}) {{
        simdgroup_matrix<{it},8,8> a0, a1;
{decl}
        for (uint kk = 0; kk < D / 8; kk++) {{
            simdgroup_load(a0, X + xrow + kk * 8, D);
            simdgroup_load(a1, X + xrow + 8 * D + kk * 8, D);
{loads}{macs}        }}
{stores}        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t = 0; t < {W//2}; t++) {{
            uint col = h * {W//2} + t, c = c0 + col;
            float a = csq[c] - 2 * T[r * {W} + col];
            bool lt = a < best; best = select(best, a, lt); bl = select(bl, c, lt);
        }}
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float ob = simd_shuffle_xor(best, 1); uint ol = simd_shuffle_xor(bl, 1);
    bool take = (ob < best) || (ob == best && ol < bl);
    bl = select(bl, ol, take);
    if (h == 0) labels[rbase + r] = bl;
"""

def med(f, r=3):
    mx.eval(f()); ts = []
    for _ in range(r):
        t = time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter() - t)
    return float(np.median(ts))

G = read_fvecs(ROOT / "data/gist/gist_base.fvecs")[:500_000]
n, k, SGPG = len(G), 1024, 4
amin = core._kernel("kmeans_approx_argmin", ["dot", "csq"], ["labels"], core._APPROX_ARGMIN_SRC)
print(f"GIST1M first {n:,} rows, first d columns, k={k}. Assignment only (labels), per pass.\n")
print(f'{"d":>5}{"exact":>9}{"matmul+argmin":>15}{"best fused":>12}{"variant":>14}{"vs current":>12}'
      f'{"labels != current":>19}{"!= exact":>10}')
for d in (256, 384, 512, 768, 960):
    X = np.ascontiguousarray(G[:, :d]); x = mx.array(X)
    parts = [x]; mx.eval(parts)
    C = core.kmeans_pp_init(parts, n, k, np.random.default_rng(0))
    for _ in range(2):
        C, _, _ = core.lloyd_step(parts, C)
    Cm = mx.array(C); csq = (Cm * Cm).sum(1); Ct = mx.contiguous(Cm.T)
    xh, Cth = x.astype(mx.float16), Ct.astype(mx.float16)
    mx.eval(Cm, csq, Ct, xh, Cth)
    ex, _ = core._nearest(x, Cm, "tiles"); mx.eval(ex)
    t_ex = med(lambda: mx.eval(core._nearest(x, Cm, "tiles")))
    per = max(1, core.DIST_BYTES // (4 * k))
    def current():
        out = []
        for r0 in range(0, n, per):
            m = min(per, n - r0)
            lab = amin(inputs=[x[r0:r0+m] @ Ct, csq], template=[("K", k)], grid=(m*32,1,1),
                       threadgroup=(256,1,1), output_shapes=[(m,)], output_dtypes=[mx.uint32])[0]
            mx.eval(lab); out.append(lab)
        return out[0] if len(out) == 1 else mx.concatenate(out)
    cur = current(); mx.eval(cur)
    t_cur = med(current)
    best = (1e9, None, None)
    for it, xin, ctin in (("float", x, Ct), ("half", xh, Cth)):
        for cpt in (1, 2, 4):
            kern = mx.fast.metal_kernel(name=f"cf_{it}_{cpt}", input_names=["X", "Ct", "csq"],
                                        output_names=["labels"], source=src(cpt, it))
            f = lambda kern=kern, xin=xin, ctin=ctin: kern(
                inputs=[xin, ctin, csq], template=[("D", d), ("K", k), ("SGPG", SGPG)],
                grid=((n // 16) * 32, 1, 1), threadgroup=(SGPG * 32, 1, 1),
                output_shapes=[(n,)], output_dtypes=[mx.uint32])[0]
            try:
                lab = f(); mx.eval(lab)
                t = med(f)
            except Exception as e:
                print(f"   d={d} {it} cpt={cpt}: {str(e)[:80]}"); continue
            if t < best[0]:
                best = (t, f"{it} cpt={cpt}", lab)
    t, var, lab = best
    print(f'{d:>5}{t_ex*1e3:>8.1f}m{t_cur*1e3:>14.1f}m{t*1e3:>11.1f}m{var:>14}{t_cur/t:>11.2f}x'
          f'{int(mx.sum(lab != cur)):>19,}{int(mx.sum(lab != ex)):>10,}')
    del X, x, parts, xh, Cth
