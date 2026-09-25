"""Reuse across arrays: fit here, predict there.

The library keeps per-dataset state between passes (a float16 copy, the cascade's prefix and last labels, the
tiles1 verdict). This checks that a second array of the same shape - the ordinary train/test or batched-predict
pattern, and the sliced-input pattern that shares a first part - is never served the previous array's state,
and that the input checks raise instead of aborting. Exit status 1 on any failure.
"""
import gc, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mlx.core as mx
from mlx_kmeans import KMeans, core

ok = True
def report(passed, name, detail=""):
    global ok
    ok &= passed
    print(f"{'PASS' if passed else 'FAIL'}  {name:58s} {detail}")

def wrong(X, C, lab, tol=1e-5):
    """Rows whose label is farther than the float64 nearest centre by more than a float32 tie.

    The failures this suite guards against are 3-97% of the rows at 10-40% excess distance; float32 ties at
    1e-7 are the library's documented tolerance and must not count.
    """
    X64, C64 = np.asarray(X, np.float64), np.asarray(C, np.float64)
    csq = (C64 * C64).sum(1)
    n, bad, worst = len(X64), 0, 0.0
    for s in range(0, n, 20000):
        x = X64[s:s + 20000]
        d2 = (x * x).sum(1)[:, None] + csq[None] - 2 * x @ C64.T
        excess = d2[np.arange(len(x)), lab[s:s + 20000]] / np.maximum(d2.min(1), 1e-300) - 1
        bad += int(np.sum(excess > tol)); worst = max(worst, float(excess.max()))
    return bad, worst

def fresh():
    core._approx_state.clear(); core._cascade_state.clear(); core._tiles1_slow.clear(); core._bounds_state.clear()

rng = np.random.default_rng(7)
n = 40_000
for name, d, k, exact in [("fused  d=8   k=16", 8, 16, True), ("tiles1 d=48  k=64", 48, 64, True),
                          ("cascade d=128 k=256", 128, 256, True), ("approx d=256 k=64", 256, 64, False),
                          ("approx d=64  k=32", 64, 32, False), ("bounds d=512 k=64", 512, 64, True)]:
    fresh()
    A = rng.standard_normal((n, d)).astype(np.float32)
    iters = 6 if d >= 512 else 3                              # enough steps for the bounds path to engage
    m = KMeans(n_clusters=k, max_iter=iters, random_state=0, exact=exact).fit(A)
    if d >= 512:                                            # the estimator's own outputs, through the bounds kernel
        bad, worst = wrong(A, m.cluster_centers_, m.labels_)
        A64, C64 = A.astype(np.float64), m.cluster_centers_.astype(np.float64)
        ref_inertia = float(((A64 - C64[m.labels_]) ** 2).sum())
        report(bad == 0 and abs(m.inertia_ - ref_inertia) <= 1e-6 * ref_inertia, f"fit labels_/inertia_        {name}",
               f"{bad} wrong, inertia rel err {abs(m.inertia_ - ref_inertia) / ref_inertia:.1e}")
    del A; gc.collect()                                     # free the id for the next array to take
    report(not (core._approx_state or core._cascade_state or core._bounds_state),
           f"state released with the arrays {name}",
           f"approx {len(core._approx_state)}, cascade {len(core._cascade_state)}, bounds {len(core._bounds_state)}")
    B = (rng.standard_normal((n, d)) * 1.3 + 0.5).astype(np.float32)
    lab = m.predict(B)
    bad, worst = wrong(B, m.cluster_centers_, lab)
    limit = 0 if exact else 0.1 * n                          # a stale fp16 copy gives 97% wrong; fp16 itself ~0.1%
    report(bad <= limit, f"predict on a second array   {name}", f"{bad} wrong of {n}, worst excess {worst:.1e}")
    # a second fit of the same shape, against a fit of the same data from a clean state (bit-identical, as documented)
    m2 = KMeans(n_clusters=k, max_iter=iters, random_state=0, exact=exact).fit(B)
    fresh()
    m3 = KMeans(n_clusters=k, max_iter=iters, random_state=0, exact=exact).fit(B.copy())
    report(m2.inertia_ == m3.inertia_ and np.array_equal(m2.cluster_centers_, m3.cluster_centers_),
           f"refit on a second array     {name}", f"inertia {m2.inertia_:.6g} vs clean {m3.inertia_:.6g}")

# MLX-array inputs the caller keeps alive: the first array's state must not serve the second (approx, the 97%-wrong case)
fresh()
A = mx.array(rng.standard_normal((n, 256)).astype(np.float32)); mx.eval(A)
m = KMeans(n_clusters=64, max_iter=3, random_state=0, exact=False).fit(A)
B = mx.array((rng.standard_normal((n, 256)) * 1.3 + 0.5).astype(np.float32)); mx.eval(B)
bad, worst = wrong(np.array(B), m.cluster_centers_, m.predict(B))
report(bad <= 0.1 * n, "predict on a second live MLX array (approx)", f"{bad} wrong of {n}, state entries {len(core._approx_state)}")
del A, B; gc.collect()
report(not core._approx_state, "approx state released with the MLX arrays", f"{len(core._approx_state)} entries")

# batched predict, every batch the same shape, one of them rescaled: the cascade's prefix bounds from the previous
# batch are wrong for this one, and a stale entry mislabels or raises
fresh()
T = rng.standard_normal((60_000, 128)).astype(np.float32)
m = KMeans(n_clusters=256, max_iter=5, random_state=0).fit(T)
bad, worst, raised = 0, 0.0, None
for i in range(4):
    b = (rng.standard_normal((15_000, 128)) * (0.1 if i == 2 else 1.0)).astype(np.float32)
    try:
        lab = m.predict(b)
    except ValueError as e:
        raised = str(e)[:50]; continue
    nb, w = wrong(b, m.cluster_centers_, lab)
    bad += nb; worst = max(worst, w)
report(bad == 0 and raised is None, "batched predict, one batch rescaled (cascade)",
       f"{bad} wrong, worst excess {worst:.2e}" + (f", raised: {raised}" if raised else ""))

# sliced inputs that share their first part
fresh()
P, Q, R = (mx.array(rng.standard_normal((20_000, 128)).astype(np.float32)) for _ in range(3)); mx.eval(P, Q, R)
m = KMeans(n_clusters=256, max_iter=3, random_state=0).fit([P, Q])
bad, worst = wrong(np.concatenate([np.array(P), np.array(R)]), m.cluster_centers_, m.predict([P, R]))
report(bad == 0, "predict on slices sharing the first part (cascade)", f"{bad} wrong, worst excess {worst:.1e}")

# input checks: raise, never abort
def raises(f):
    try: f(); return None
    except ValueError as e: return str(e)
m = KMeans(n_clusters=8, max_iter=2, random_state=0).fit(rng.standard_normal((5000, 8)).astype(np.float32))
e = raises(lambda: m.predict(rng.standard_normal((300, 16)).astype(np.float32)))
report(e is not None and "features" in e, "predict with a different feature count", (e or "no error")[:60])
X = mx.array(rng.standard_normal((3000, 12)).astype(np.float32)); C = np.zeros((100, 12), np.float32)
e = raises(lambda: core.assign_mlx([X], C, method="tiles"))
report(e is not None and "tiles" in e, "method='tiles' with k % 8 != 0", (e or "no error")[:60])
saved = core.PART_MAX_ELEMENTS
try:
    core.PART_MAX_ELEMENTS = 1_000_000
    parts = KMeans._parts(mx.array(np.zeros((2500, 1000), np.float32)))
    report(len(parts) == 3 and sum(len(p) for p in parts) == 2500, "MLX input over the element limit is sliced", f"{len(parts)} parts")
    e = raises(lambda: core.assign_mlx([mx.array(np.zeros((2500, 1000), np.float32))], np.zeros((8, 1000), np.float32)))
    report(e is not None and "32-bit" in e, "a single part over the element limit is refused", (e or "no error")[:60])
finally:
    core.PART_MAX_ELEMENTS = saved
print("all passed" if ok else "FAILURES"); sys.exit(0 if ok else 1)
