"""How much of each Lloyd iteration's n*k distance work is actually needed, on GIST 300k x 960, k=1024?
Per iteration, on a fixed 20k-row sample with the full old-distance matrix:
  changed   = points whose label changes under the new centers
  hamerly   = points whose margin (d2-d1) >= drift[own] + max drift  -> need NO distance work at all
  elkan     = fraction of the n*k distances that must be recomputed: centers j with d_old(x,j) - drift[j] < d1 + drift[own]
"""
import sys, time, numpy as np, mlx.core as mx
sys.path.insert(0, "/Users/satya/git/kmeans")
from mlx_kmeans import core
n, k, iters, ns = 300_000, 1024, 14, 20_000
raw = np.fromfile("data/gist/gist_learn.fvecs", dtype=np.int32, count=n * 961).reshape(n, 961)
X = raw[:, 1:].view(np.float32).copy(); del raw
rng = np.random.default_rng(0)
C = X[rng.choice(n, k, replace=False)].astype(np.float32)          # random-row init (more drift than k-means++: conservative)
parts = [mx.array(X)]; mx.eval(parts[0])
sidx = np.sort(rng.choice(n, ns, replace=False)); Sx = mx.array(X[sidx]); sn2 = mx.sum(Sx * Sx, axis=1, keepdims=True); mx.eval(Sx, sn2)
def dists(Cm):
    Cm = mx.array(Cm); D = sn2 + mx.sum(Cm * Cm, axis=1)[None, :] - 2 * (Sx @ Cm.T)
    return mx.sqrt(mx.maximum(D, 0.0))
print(f"{'it':>2} {'inertia':>12} {'ms':>6} {'chg%':>6} {'maxdrift':>8} {'meddrift':>8} {'hamerly-skip%':>13} {'elkan-needed%':>13}")
Dold = dists(C); mx.eval(Dold)
for it in range(iters):
    t = time.perf_counter(); Cn, inertia, nempty = core.lloyd_step(parts, C); ms = (time.perf_counter() - t) * 1e3
    drift = np.linalg.norm(Cn.astype(np.float64) - C.astype(np.float64), axis=1).astype(np.float32)
    Dn = dists(Cn); mx.eval(Dn)
    lab_old = np.array(mx.argmin(Dold, axis=1)); lab_new = np.array(mx.argmin(Dn, axis=1))
    Do = np.array(Dold); srt = np.sort(Do, axis=1); d1, d2 = srt[:, 0], srt[:, 1]
    u = d1 + drift[lab_old]                                          # upper bound on the new distance to the old center
    hamerly = np.mean(d2 - drift.max() >= u)
    needed = (Do - drift[None, :] < u[:, None])                      # centers that could beat the bound -> must be recomputed
    elkan = needed.mean()
    print(f"{it:>2} {inertia:>12.4e} {ms:>6.1f} {100*np.mean(lab_old != lab_new):>6.2f} {drift.max():>8.4f} {np.median(drift):>8.4f} {100*hamerly:>13.1f} {100*elkan:>13.2f}")
    C, Dold = Cn, Dn
