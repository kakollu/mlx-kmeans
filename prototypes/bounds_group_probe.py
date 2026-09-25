"""Group bounds (Yinyang-style, groups of G centres) vs per-centre bounds, GIST 300k x 960, k=1024.
For each iteration: fraction of points passing the group filter untouched (no centre distances at all, after the
own-centre distance is recomputed), and the fraction of n*k centre distances visited = sum of visited group sizes."""
import sys, time, numpy as np, mlx.core as mx
sys.path.insert(0, "/Users/satya/git/kmeans")
from mlx_kmeans import core
n, k, iters, ns, G = 300_000, 1024, 12, 20_000, int(sys.argv[1]) if len(sys.argv) > 1 else 8
raw = np.fromfile("data/gist/gist_learn.fvecs", dtype=np.int32, count=n * 961).reshape(n, 961)
X = raw[:, 1:].view(np.float32).copy(); del raw
rng = np.random.default_rng(0)
C = X[rng.choice(n, k, replace=False)].astype(np.float32)
parts = [mx.array(X)]; mx.eval(parts[0])
sidx = np.sort(rng.choice(n, ns, replace=False)); Sx = mx.array(X[sidx]); sn2 = mx.sum(Sx * Sx, axis=1, keepdims=True); mx.eval(Sx, sn2)
def dists(Cm):
    Cm = mx.array(Cm); D = sn2 + mx.sum(Cm * Cm, axis=1)[None, :] - 2 * (Sx @ Cm.T)
    return np.array(mx.sqrt(mx.maximum(D, 0.0)))
def coherent_order(C, t):
    """order centres so that contiguous blocks are spatially coherent: k-means on the centres, sort by group."""
    r = np.random.default_rng(1); M = C[r.choice(len(C), t, replace=False)].astype(np.float64); C64 = C.astype(np.float64)
    for _ in range(10):
        g = np.argmin((C64 * C64).sum(1)[:, None] + (M * M).sum(1)[None] - 2 * C64 @ M.T, axis=1)
        for j in range(t):
            if np.any(g == j): M[j] = C64[g == j].mean(0)
    return np.argsort(g, kind="stable")
orders = {"random blocks": np.arange(k), "coherent blocks": coherent_order(C, k // G)}
t = (k + G - 1) // G
print(f"groups of {G} -> t={t}; lb memory at 300k rows: {ns and 300_000 * t * 4 / 1e6:.0f} MB fp32")
print(f"{'it':>2} {'chg%':>6} | {'percentre%':>10} | " + " | ".join(f"{name:>16}: skip% visit%" for name in orders))
Dold = dists(C)
for it in range(iters):
    Cn, inertia, nempty = core.lloyd_step(parts, C)
    drift = np.linalg.norm(Cn.astype(np.float64) - C.astype(np.float64), axis=1).astype(np.float32)
    Dn = dists(Cn); lab_old = np.argmin(Dold, axis=1); lab_new = np.argmin(Dn, axis=1)
    ub = Dn[np.arange(ns), lab_old]                                  # tight: own distance recomputed exactly
    percentre = np.mean(Dold - drift[None, :] < ub[:, None])
    cols = []
    for name, order in orders.items():
        inv = np.empty(k, np.int64); inv[order] = np.arange(k)      # internal index of each caller centre
        Dg = Dold[:, order]; dg = drift[order]                       # in internal order; group g = columns gG..gG+G-1
        own_int = inv[lab_old]
        lbg = np.full((ns, t), np.inf, np.float32); Dm = Dg.copy(); Dm[np.arange(ns), own_int] = np.inf
        for g in range(t):
            lbg[:, g] = Dm[:, g * G:(g + 1) * G].min(1) - dg[g * G:(g + 1) * G].max()
        need = lbg < ub[:, None]
        cols.append(f"{name:>16}: {100 * np.mean(~need.any(1)):5.1f} {100 * need.sum() * G / (ns * k):6.2f}")
    print(f"{it:>2} {100 * np.mean(lab_old != lab_new):6.2f} | {100 * percentre:10.2f} | " + " | ".join(cols))
    C, Dold = Cn, Dn
