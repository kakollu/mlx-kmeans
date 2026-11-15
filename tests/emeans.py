#!/usr/bin/env python3
"""e-means tests (mlx_kmeans/emeans.py, docs/e-means.md).

Three parts, by what they need:

  --numpy     NumPy only, imports no MLX. The scaling laws against hand-computed values (including the worked
              checks in the paper paraphrase), the cosine schedule, the revival draw, the shared permutation, and
              self-checks of the float64 reference: it reaches the float64 Lloyd loss from the same start on
              overlapping blobs in 3 epochs, its last-epoch mean is the direct mean over its recorded
              assignments, one full batch at a0 = 1 is one Lloyd step, and driven by its own labels it
              reproduces itself.
  --mlx-cpu   Plain MLX ops on the CPU device (no Metal kernels): the GPU fit's plumbing (gather, moving
              averages, revival, last-epoch compensated sums, trace, determinism, slicing) driven by a batch step
              written in MLX ops, against the reference; the GPU-planned segment layout against a NumPy
              emulation of the segment kernel; the two-sum accumulator.
  (default)   All of the above, then the GPU part (needs the Metal kernels): the fit against the reference on
              seeded cases for every batch method and accumulator (each case asserts the method and accumulator
              it ran, and a tiles1 chosen by auto is checked against its probe), revival, a0 = 1, slices,
              k-means++ init, uncentred data, the approximate path in float16 and float32, the one-hot
              fallback; determinism; KMeans(algorithm="emeans") end to end; per-dataset caches.

The per-batch criteria are accuracy.py's: every label float64-optimal to within 8 eps32 (|x|^2 + |c|^2) of the
nearest centre for the centres the batch was assigned against; counts exact; batch sums within 1e-6 of the
float64 sums, relative to the data's RMS coordinate (per cluster mean). One moving-average step, from the fit's own
state, within 1e-6 of the RMS coordinate. Each revival decision is exactly the shared draw on the N the fit read.
Against the reference driven by the same labels and revival decisions: the trajectory within 1e-5, the last-epoch
means within 1e-6 (check_fit's docstring has the details). The reference's own float64 revival decisions are
reported, not judged.

Run:  python3 tests/emeans.py [--numpy | --mlx-cpu]        (exit code 1 on any failure)
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_kmeans import emeans as em  # noqa: E402  (imports no MLX)

EPS32 = float(np.finfo(np.float32).eps)
RESULTS = []


def report(ok, name, detail=""):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name:58s} {detail}", flush=True)


def blobs(rng, n, d, true_k, spread, noise):
    centers = rng.uniform(-spread, spread, size=(true_k, d))
    return (centers[rng.integers(0, true_k, n)] + rng.normal(0, noise, (n, d))).astype(np.float32)


def lloyd64(X64, C, iters=300):
    """Float64 Lloyd to convergence (labels stop changing), empty clusters keep their centre."""
    k = len(C)
    for it in range(iters):
        lab, _ = em._nearest64(X64, C)
        cnt = np.bincount(lab, minlength=k)
        sums = np.zeros_like(C)
        np.add.at(sums, lab, X64)
        new = C.copy()
        has = cnt > 0
        new[has] = sums[has] / cnt[has, None]
        if np.array_equal(new, C):
            break
        C = new
    return C, float(em._nearest64(X64, C)[1].mean()), it + 1


# ---------------------------------------------------------------- NumPy part

def numpy_part():
    print("== NumPy: laws, schedule, draws, reference self-checks")
    # Hand-computed: B = 2^floor(log2(B_raw) + 0.5) capped at n, B_raw = 0.035 k^0.13 n^0.88 T^0.81 (/8 above 2M rows);
    # a0 = min(0.824 k^-0.46 B^0.90 n^-0.45 T^-0.11, 1) at that B.
    for n, k, T, B_exp, a_exp, note in [
            (1_000_000, 16384, 3, 65536, 0.3628, "paper-spec check (a), M = 45"),
            (1_000_000, 1024, 25, 262144, 1.0, "check (b): raw a0 3.58, clipped to 1"),
            (1_000_000, 1024, 2, 32768, 0.7277, "check (c)"),
            (1_000_000, 4096, 3, 65536, 0.6864, "check (d): log2 rounding (linear gives 32768)"),
            (1_000_000, 1024, 3, 32768, 0.6960, "sift-k1024 shape"),
            (10_000_000, 65536, 3, 65536, 0.0680, "n > 2M: /8 before rounding (else 524288)"),
            (2_000_000, 1024, 3, 65536, None, "n = 2M exactly: not divided"),
            (2_000_001, 1024, 3, 8192, None, "n = 2M + 1: divided by 8"),
            (1000, 1000, 100, 1000, 0.4634, "capped at n (law rounds to 2048)"),
            (20_000, 16, 3, 1024, 1.0, "small case: raw a0 1.21, clipped")]:
        B = em.batch_size(n, k, T)
        a = em.step0(n, k, T, B)
        ok = B == B_exp and (a_exp is None or abs(a - a_exp) <= 5e-4)
        report(ok, f"laws n={n:,} k={k} T={T}", f"B {B} (want {B_exp}), a0 {a:.4f}" +
               (f" (want {a_exp})" if a_exp is not None else "") + f"  [{note}]")
    raw = lambda n, k, T: 0.035 * math.exp(0.13 * math.log(k) + 0.88 * math.log(n) + 0.81 * math.log(T))  # noqa: E731
    worked = [(raw(1e6, 16384, 3), 5.73e4), (raw(1e6, 1024, 25), 2.23e5), (raw(1e6, 1024, 2), 2.88e4),
              (raw(1e6, 4096, 3), 4.78e4)]
    a_raw_b = 0.824 * 1024 ** -0.46 * 262144 ** 0.90 * 1e6 ** -0.45 * 25 ** -0.11
    ok = all(abs(v / w - 1) < 5e-3 for v, w in worked) and abs(a_raw_b - 3.58) < 0.01
    report(ok, "raw law values match the paper-spec worked checks",
           ", ".join(f"{v:.4g}" for v, _ in worked) + f"; raw a0 (b) {a_raw_b:.3f}")
    P = em.plan(1_000_000, 16384, 3)
    report(P == dict(B=65536, a0=em.step0(1_000_000, 16384, 3, 65536), per_epoch=15, M=45), "plan(1e6, 16384, 3)",
           f"{P}")
    P2 = em.plan(1_000_000, 16384, 3, batch_size_=10_000)
    report(P2["B"] == 10_000 and abs(P2["a0"] - em.step0(1_000_000, 16384, 3, 10_000)) < 1e-15 and P2["M"] == 300,
           "plan with batch_size given: a0 from the law at that B", f"{P2}")
    bad = 0
    for args in [(0, 4, 3), (10, 0, 3), (10, 4, 0), (10, 4, 2.5), (True, 4, 3)]:
        try:
            em.batch_size(*args)
            bad += 1
        except ValueError:
            pass
    for kw in [dict(batch_size_=11), dict(step0_=0.0), dict(step0_=1.5)]:
        try:
            em.plan(10, 4, 3, **kw)
            bad += 1
        except ValueError:
            pass
    report(bad == 0, "laws and plan refuse bad inputs", f"{bad} accepted")

    # schedule: a(0) = a0, halfway 0.55 a0, decreasing, last batch just above 0.1 a0
    M, a0 = 45, 0.3628
    s = np.array([em.schedule(t, M, a0) for t in range(M)])
    ok = (abs(s[0] - a0) < 1e-15 and np.all(np.diff(s) < 0) and 0.1 * a0 < s[-1] < 0.102 * a0
          and abs(em.schedule(M / 2, M, a0) - 0.55 * a0) < 1e-15)
    report(ok, "cosine schedule a0 -> 0.1 a0 over M batches", f"a(0) {s[0]:.4f}, a(M-1) {s[-1]:.5f} = {s[-1] / a0:.4f} a0")

    # revival draw
    seed = 5
    d0, o0 = em._revive_plan(np.full(8, 3.0), seed, 4)
    N = np.array([10.0, 10.0, 0.5, 10.0, 0.0, 9.0])               # mean 6.58, threshold 0.658: 2 and 4 are dead
    d1, o1 = em._revive_plan(N, seed, 9)
    d2, o2 = em._revive_plan(N, seed, 9)
    d3, o3 = em._revive_plan(np.array([100.0, 0.0, 0.0, 0.0]), seed, 14)   # more dead than live donors
    ok = (len(d0) == 0 and list(d1) == [2, 4] and len(set(o1)) == 2 and set(o1) <= {0, 1, 3, 5}
          and np.array_equal(o1, o2) and list(d3) == [1] and list(o3) == [0])
    report(ok, "revival: dead set, distinct live donors, lowest dead first", f"dead {d1.tolist()} donors {o1.tolist()}; "
           f"1 live: dead {d3.tolist()} donor {o3.tolist()}")
    Nw = np.array([1.0, 2.0, 7.0, 0.0])                            # one dead (index 3): the donor is drawn in proportion to N
    first = np.array([em._revive_plan(Nw, seed, t)[1][0] for t in range(20000)])
    freq = np.bincount(first, minlength=3)[:3] / len(first)
    report(np.all(np.abs(freq - Nw[:3] / Nw.sum()) < 0.015), "revival: donor frequency proportional to N",
           f"{np.round(freq, 3)} vs {Nw[:3] / Nw.sum()}")
    Nm = np.array([4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0])            # three dead: donors without replacement
    picks = [tuple(em._revive_plan(Nm, seed, t)[1]) for t in range(4000)]
    report(all(len(set(p)) == 3 for p in picks), "revival: donors drawn without replacement", f"{len(picks)} rounds")

    # shared permutation
    n, B = 10_007, 1000
    b1, b2, b3 = em._epoch_batches(1, 0, n, B), em._epoch_batches(1, 0, n, B), em._epoch_batches(1, 1, n, B)
    flat = b1.reshape(-1)
    ok = (b1.shape == (10, 1000) and np.array_equal(b1, b2) and not np.array_equal(b1, b3)
          and np.all(np.diff(b1.astype(np.int64), axis=1) > 0) and len(np.unique(flat)) == 10_000 and flat.max() < n)
    report(ok, "epoch batches: seeded, sorted, distinct, remainder dropped", f"{b1.shape}, {n - flat.size} dropped")
    # the documented stream: default_rng([seed, 2, e]).permutation(n), cut, sorted (built from a uint32 shuffle)
    same = True
    for n_, B_, sd, e_ in [(10_007, 1000, 1, 0), (65_536, 4096, 0, 2), (1_000_003, 32_768, 7, 1), (5, 5, 3, 0)]:
        per_ = n_ // B_
        want = np.sort(np.random.default_rng([sd, 2, e_]).permutation(n_)[:per_ * B_].reshape(per_, B_), axis=1)
        got = em._epoch_batches(sd, e_, n_, B_)
        same &= got.dtype == np.uint32 and np.array_equal(got, want)
    report(same, "epoch batches equal the sorted cuts of default_rng([seed, 2, e]).permutation(n)",
           f"NumPy {np.__version__}")
    r1, r2 = em._init_rows(3, n, 50), em._init_rows(3, n, 50)
    report(np.array_equal(r1, r2) and len(set(r1)) == 50, "init rows: seeded and distinct")

    # reference self-checks
    rng = np.random.default_rng(11)
    X = blobs(rng, 30_000, 32, 32, 3, 1.0)                         # overlapping blobs, k = 4x the blob count
    X64 = X.astype(np.float64)
    ratios, revived = [], []
    for sd in (0, 1, 2):
        C, rec = em.emeans_reference(X, 128, epochs=3, seed=sd)
        le = float(em._nearest64(X64, C)[1].mean())
        _, ll, its = lloyd64(X64, X64[em._init_rows(sd, len(X), 128)].copy())
        ratios.append(le / ll)
        revived.append(sum(rec["revived"]))
    report(max(ratios) <= 1.01 and float(np.median(ratios)) <= 1.005,
           "reference reaches float64 Lloyd's loss in 3 epochs (<= 1%)",
           f"e-means / Lloyd-to-convergence from the same rows: {', '.join(f'{r:.4f}' for r in ratios)}; "
           f"revived {revived}")

    trace = []
    C, rec = em.emeans_reference(X, 128, epochs=3, seed=4, trace=trace)
    lab = rec["last_labels"]
    keep = lab >= 0
    cnt = np.bincount(lab[keep], minlength=128)
    sums = np.zeros((128, 32))
    np.add.at(sums, lab[keep], X64[keep])
    used = cnt > 0
    err = float(np.max(np.abs(C[used] - sums[used] / cnt[used, None])))
    kept_ema = np.array_equal(C[~used], trace[-1]["C"][~used])
    ok = err <= 1e-12 and kept_ema and keep.sum() == rec["batches_per_epoch"] * rec["B"]
    report(ok, "reference: last-epoch mean = direct mean of its assignments",
           f"max diff {err:.1e}; {int(keep.sum())} rows ({rec['dropped_per_epoch']} dropped), "
           f"{int((~used).sum())} empty keep the EMA centre")

    C0 = X64[:128].copy()
    one, _ = em.emeans_reference(X, 128, epochs=1, seed=0, batch_size=len(X), step0=1.0, init=C0)
    lab0, _ = em._nearest64(X64, C0)
    cnt0 = np.bincount(lab0, minlength=128)
    s0 = np.zeros((128, 32))
    np.add.at(s0, lab0, X64)
    step = C0.copy()
    step[cnt0 > 0] = s0[cnt0 > 0] / cnt0[cnt0 > 0, None]
    report(float(np.max(np.abs(one - step))) <= 1e-12, "reference: one full batch at a0 = 1 is one Lloyd step",
           f"max diff {float(np.max(np.abs(one - step))):.1e}")

    driven, _ = em.emeans_reference(X, 128, epochs=3, seed=4, labels=[tr["labels"] for tr in trace])
    report(np.array_equal(driven, C), "reference driven by its own labels reproduces itself")
    chk = [tr for tr in trace if tr["N_check"] is not None]
    revs = [(tr["dead"], tr["donors"]) for tr in chk]
    driven2, _ = em.emeans_reference(X, 128, epochs=3, seed=4, labels=[tr["labels"] for tr in trace], revivals=revs)
    ok = (np.array_equal(driven2, C) and len(chk) == rec["M"] // 5 and all(tr["t"] % 5 == 4 for tr in chk)
          and all(np.array_equal(em._revive_plan(tr["N_check"], 4, tr["t"])[1], tr["donors"]) for tr in chk))
    # a decision handed in is the one applied: swap the first two donors of the first check that revived two or more
    j = next(i for i, (dd, _) in enumerate(revs) if len(dd) >= 2)
    swapped = list(revs)
    swapped[j] = (revs[j][0], revs[j][1][[1, 0] + list(range(2, len(revs[j][1])))])
    t3 = []
    em.emeans_reference(X, 128, epochs=3, seed=4, labels=[tr["labels"] for tr in trace], revivals=swapped, trace=t3)
    dd, oo = swapped[j]
    tj = chk[j]["t"]
    p_, q_ = em._split_factors(32)
    # both children come from the same donor centre: c_dead / p = c_donor / q, for the donor handed in
    ok &= (np.array_equal(t3[tj]["donors"], oo) and
           np.allclose(t3[tj]["C"][dd[0]] / p_, t3[tj]["C"][oo[0]] / q_, rtol=1e-12, atol=0) and
           not np.allclose(t3[tj]["C"][dd[0]] / p_, t3[tj]["C"][oo[1]] / q_, rtol=1e-6, atol=0))
    report(ok, "reference: driven by its own labels and revivals reproduces itself; handed decisions are applied",
           f"{len(chk)} checks, revived {sum(len(dd) for dd, _ in revs)}")

    # revival with centres no row is near: at a0 = 1 their N becomes exactly 0 after the first batch and they keep
    # their centre (choice 4) until the 5th batch revives them
    Xs = blobs(rng, 20_000, 16, 16, 10, 1.0)
    init = Xs[em._init_rows(9, len(Xs), 16)].astype(np.float64)
    init[:3] = 1000.0
    tr2 = []
    Cs, rec2 = em.emeans_reference(Xs, 16, epochs=3, seed=9, init=init, trace=tr2)
    zero_kept = np.all(tr2[0]["N"][:3] == 0) and np.array_equal(tr2[0]["C"][:3], init[:3])
    revived_first = tr2[4]["dead"] is not None and set(tr2[4]["dead"]) >= {0, 1, 2}
    ok = rec2["a0"] == 1.0 and zero_kept and revived_first and float(np.abs(Cs).max()) < 100
    report(ok, "reference: a0 = 1, empty centres kept, revived at batch 5",
           f"a0 {rec2['a0']}, revived {rec2['revived'][:3]}..., max |c| after {float(np.abs(Cs).max()):.1f}")


# ---------------------------------------------------------------- the fit against the reference (MLX CPU and GPU)

def make_parts(X, per=None):
    import mlx.core as mx
    per = per or len(X)
    return [mx.array(X[s:s + per]) for s in range(0, len(X), per)]


def label_check(X64, rows, labels, C_before):
    """-> (wrong, ties): labels farther than accuracy.py's float32 window from the float64 nearest of C_before."""
    x = X64[rows]
    C = C_before.astype(np.float64)
    ref, ref_d = em._nearest64(x, C)
    d_lab = ((x - C[labels]) ** 2).sum(1)
    tol = 8 * EPS32 * ((x ** 2).sum(1) + (C[labels] ** 2).sum(1))
    wrong = int((d_lab > ref_d + tol).sum())
    return wrong, int((labels != ref).sum()) - wrong


def ema_step_err(trace, k, scale):
    """One moving-average step at a time, in float64 from the fit's own float32 state before the step.

    S before a step is taken as C_before * N_before (C = fl(S / N), so this is S to float32 rounding). Clusters
    a revival touched at that step are left to the driven comparison.
    -> (centre err / scale, N err / max(N_j, mean N), N err / mean N).

    The N error is judged per cluster against max(N_j, mean N): the float32 rounding of N_j is relative to N_j
    itself (N_new >= each of its two non-negative terms, so a correct step is within a few 2^-24 of N_j), and a
    cluster holding a large share of the batch would otherwise fail on its own ulp when divided by mean N.
    At or below the mean the scale is mean N, as before. The mean-N value is returned for the report.
    """
    c_err = n_err = n_err_mean = 0.0
    N_prev = np.full(k, np.float32(em.INIT_COUNT), dtype=np.float64)
    for tr in trace:
        a = tr["a"]
        Cb = tr["C_before"].astype(np.float64)
        N_new = (1.0 - a) * N_prev + a * tr["counts"]
        S_new = (1.0 - a) * (Cb * N_prev[:, None]) + a * tr["sums"]
        has = N_new > 0
        C_exp = np.where(has[:, None], S_new / np.where(has, N_new, 1.0)[:, None], Cb)
        keep = np.ones(k, bool)
        if tr["dead"] is not None and len(tr["dead"]):
            keep[tr["dead"]] = keep[tr["donors"]] = False
        if keep.any():
            c_err = max(c_err, float(np.max(np.abs(tr["C"][keep].astype(np.float64) - C_exp[keep]))) / scale)
            dN = np.abs(tr["N"][keep] - N_new[keep])
            m = max(float(N_new.mean()), 1e-30)
            n_err = max(n_err, float(np.max(dN / np.maximum(N_new[keep], m))))
            n_err_mean = max(n_err_mean, float(np.max(dN)) / m)
        N_prev = tr["N"].astype(np.float64)
    return c_err, n_err, n_err_mean


def revive_agreement(trace, rtrace, seed):
    """The reference's own float64 revival decisions against the fit's float32 ones -> (disagreements, detail).

    Reported, not judged: the two N differ by float32 rounding and by the trajectory's float32 drift, so a cluster
    within that distance of 0.1 mean(N) can land on the other side, and two donor keys (log N + Gumbel noise)
    within it can swap order. At a disagreement the detail gives how close it was: |N - thr| / thr for a dead set
    that differs, the key gap for a donor order that differs.
    """
    bad, worst = 0, []
    for g, r in zip(trace, rtrace):
        if g["N_check"] is None:
            continue
        dd, oo = em._revive_plan(r["N_check"], seed, g["t"])
        if np.array_equal(dd, g["dead"]) and np.array_equal(oo, g["donors"]):
            continue
        bad += 1
        Nr = r["N_check"]
        thr = em.REVIVE_TAU * Nr.mean()
        if not np.array_equal(dd, g["dead"]):
            sd = np.setxor1d(dd, g["dead"])
            worst.append(f"t={g['t']} dead set differs, |N-thr|/thr {float(np.max(np.abs(Nr[sd] - thr))) / thr:.1e}"
                         if len(sd) else f"t={g['t']} dead set order differs")
        else:
            u = np.random.default_rng([seed, 3, g["t"]]).random(len(Nr))
            with np.errstate(divide="ignore"):
                key = np.log(Nr) - np.log(-np.log(np.maximum(u, np.finfo(np.float64).tiny)))
            i = np.flatnonzero(oo != g["donors"])
            gap = float(np.min(np.abs(key[oo[i]] - key[g["donors"][i]])))
            worst.append(f"t={g['t']} donor order differs at {len(i)} of {len(oo)}, key gap {gap:.1e}")
    return bad, "; ".join(worst[:3])


def check_fit(name, X, k, per=None, approx=False, free=False, drift_tol=1e-5, expect=None, **kw):
    """Run emeans_fit with a trace and check every batch, then compare with the reference driven by the same labels.

    Per batch: labels (accuracy.py's window), counts exact, sums within 1e-6, and one moving-average step from the
    fit's own state within 1e-6 of the RMS coordinate (centres) and of max(N_j, mean N) (counts; see
    ema_step_err). Revival plumbing: every traced decision is exactly the shared
    draw (_revive_plan) on the float32 N the fit read. Against the reference driven by the same labels and the same
    revival decisions: the final centres within 1e-6 where the last epoch gave a cluster rows (a cluster it gave
    none keeps its moving-average centre: bit-equal to the fit's last state, and within drift_tol of the
    reference), and the centre trajectory within drift_tol, looser because the float32 state carries rounding over
    about 1/a batches (the paper keeps N and S in float32 too; App J.2). The reference's own float64 revival
    decisions are reported, not judged (revive_agreement). For the approximate path (approx=True) the label count
    is reported but not judged: labels from the matmul are taken on trust by design; counts, sums and the moving
    averages are still checked. expect: (method, accumulator) the fit must report, or (method, accumulator,
    switch) for a tiles1 chosen by auto, whose first-batch probe moves it to `switch` when the fallback rate is
    over TILES1_MAX_FALLBACK, and only then.
    """
    X64 = X.astype(np.float64)
    scale = float(np.sqrt((X64 ** 2).mean()))
    parts = make_parts(X, per)
    trace = []
    t0 = time.perf_counter()
    C, rec = em.emeans_fit(parts, k, trace=trace, **kw)
    fit_s = time.perf_counter() - t0
    seed, epochs = rec["seed"], rec["epochs"]
    n = len(X)
    wrong = ties = 0
    count_bad = 0
    sum_err = 0.0
    rows_ok = True
    for tr in trace:
        e, i = tr["epoch"], tr["t"] - tr["epoch"] * rec["batches_per_epoch"]
        rows_ok &= np.array_equal(tr["rows"], em._epoch_batches(seed, e, n, rec["B"])[i].astype(np.int64))
        w_, t_ = label_check(X64, tr["rows"], tr["labels"], tr["C_before"])
        wrong += w_
        ties += t_
        cnt = np.bincount(tr["labels"], minlength=k)
        count_bad += int(not np.array_equal(tr["counts"], cnt.astype(np.float64)))
        s = np.zeros((k, X.shape[1]))
        np.add.at(s, tr["labels"], X64[tr["rows"]])
        used = cnt > 0
        if used.any():
            sum_err = max(sum_err, float(np.max(np.abs(tr["sums"][used] - s[used]) / cnt[used, None])) / scale)
    labels_ok = approx or wrong == 0         # approx: the count is reported, not judged (float32 matmul error)
    step_err, n_err, n_err_mean = ema_step_err(trace, k, scale)
    # revival plumbing: one check per 5 batches, each decision the shared draw on the N the fit read
    checks = [tr for tr in trace if tr["N_check"] is not None]
    plumb_bad = sum(not (np.array_equal(dd, tr["dead"]) and np.array_equal(oo, tr["donors"]))
                    for tr in checks for dd, oo in [em._revive_plan(tr["N_check"], seed, tr["t"])])
    revivals = [(tr["dead"], tr["donors"]) for tr in checks]
    plumb_ok = (plumb_bad == 0 and len(checks) == rec["M"] // em.REVIVE_EVERY
                and [len(dd) for dd, _ in revivals] == rec["revived"])
    # the moving averages, revival and last-epoch mean, from the same labels and revivals in float64
    rtrace = []
    init = trace[0]["C_before"]
    Cr, rrec = em.emeans_reference(X, k, epochs=epochs, seed=seed, batch_size=rec["B"], step0=rec["a0"],
                                   init=init, labels=[tr["labels"] for tr in trace], revivals=revivals, trace=rtrace)
    c_err = max(float(np.max(np.abs(g["C"].astype(np.float64) - r["C"]))) for g, r in zip(trace, rtrace)) / scale
    own_bad, own_detail = revive_agreement(trace, rtrace, seed)

    def final_errs(Cref, counts):
        """-> (err where the last epoch gave rows, err of the kept moving-average centres, kept ones bit-equal)."""
        used = counts > 0
        diff = np.abs(C.astype(np.float64) - Cref)
        f = float(diff[used].max()) / scale if used.any() else 0.0
        kept = float(diff[~used].max()) / scale if (~used).any() else 0.0
        return f, kept, np.array_equal(C[~used], trace[-1]["C"][~used])

    final_err, kept_err, kept_same = final_errs(Cr, rrec["last_epoch_counts"])
    empty_ok = rec["last_epoch_empty"] == rrec["last_epoch_empty"]
    loss_err = max(abs(a - b) / max(b, 1e-30) for a, b in zip(rec["train_mse"], rrec["train_mse"]))
    meth_ok, expect_note = True, ""
    if expect is not None:
        want_m, want_a = expect[0], expect[1]
        rate = rec.get("tiles1_fallback_rate")
        if len(expect) > 2:                          # tiles1 by auto: the probe decides, consistently with its rate
            want_m = expect[2] if (rate is not None and rate > em.core.TILES1_MAX_FALLBACK) else want_m
            meth_ok = rate is not None
            expect_note = f" (probe fallback {rate:.3f})" if rate is not None else " (probe did not run)"
        meth_ok = meth_ok and rec["method"] == want_m and rec["accumulate"] == want_a
        if not meth_ok:
            expect_note += f" WANTED {want_m}/{want_a}"
    ok = (rows_ok and labels_ok and count_bad == 0 and sum_err <= 1e-6 and step_err <= 1e-6 and n_err <= 1e-6
          and plumb_ok and c_err <= drift_tol and final_err <= 1e-6 and kept_same and kept_err <= drift_tol
          and empty_ok and loss_err <= 1e-5 and meth_ok)
    detail = (f"{rec['method']}/{rec['accumulate']}{expect_note} B={rec['B']} a0={rec['a0']:.3f} M={rec['M']} "
              f"parts={len(parts)}: wrong labels {wrong} (float32 ties {ties}), "
              f"counts {'ok' if count_bad == 0 else f'{count_bad} BAD'}, sum err {sum_err:.1e}, "
              f"EMA step {step_err:.1e} (N {n_err:.1e}, of mean N {n_err_mean:.1e}), drift {c_err:.1e}, "
              f"final {final_err:.1e} "
              f"(kept {rec['last_epoch_empty']}: {kept_err:.1e}{'' if kept_same else ' NOT BIT-EQUAL'}), "
              f"loss {loss_err:.1e}, revived {sum(rec['revived'])} in {len(checks)} checks"
              f"{'' if plumb_ok else ' (PLUMBING BAD)'}, reference's own decisions differ at {own_bad}"
              f"{f' [{own_detail}]' if own_bad else ''}{'' if rows_ok else ', ROWS DIFFER'}, {fit_s:.1f}s")
    if free:
        # The reference with its own labels (revivals still the fit's): only float32 ties may differ. Judged up to
        # and including the first batch where a label differs; after it the two runs no longer share centres, so
        # later labels are not comparable.
        ftrace = []
        Cf, frec = em.emeans_reference(X, k, epochs=epochs, seed=seed, batch_size=rec["B"], step0=rec["a0"],
                                       init=init, revivals=revivals, trace=ftrace)
        diff = nontie = 0
        judged = 0
        for g, r in zip(trace, ftrace):
            judged += 1
            dl = np.flatnonzero(g["labels"] != r["labels"])
            if len(dl):
                diff = len(dl)
                x = X64[g["rows"][dl]]
                Cb = r["C_before"]
                dg, dr = ((x - Cb[g["labels"][dl]]) ** 2).sum(1), ((x - Cb[r["labels"][dl]]) ** 2).sum(1)
                nontie = int((dg > dr + 8 * EPS32 * ((x ** 2).sum(1) + (Cb[g["labels"][dl]] ** 2).sum(1))).sum())
                break
        f_err, f_kept, _ = final_errs(Cf, frec["last_epoch_counts"])
        ok &= nontie == 0 and (diff > 0 or (f_err <= 1e-6 and f_kept <= drift_tol))
        detail += (f"; free reference: {diff} labels differ at batch {judged - 1} ({nontie} not ties)" if diff else
                   f"; free reference: no label differs in {judged} batches, final {f_err:.1e} (kept {f_kept:.1e})")
    report(ok, name, detail)
    return C, rec, trace


def mlx_step(k):
    """A batch step in plain MLX ops (direct distances, scatter-add), for the CPU device: no Metal kernels."""
    import mlx.core as mx

    def step(xb, C):
        diff = xb[:, None, :] - C[None, :, :]
        d2 = (diff * diff).sum(-1)
        lab = mx.argmin(d2, axis=1).astype(mx.uint32)
        best = mx.take_along_axis(d2, lab.astype(mx.int32)[:, None], axis=1)[:, 0]
        vals = mx.concatenate([mx.ones((xb.shape[0], 1), dtype=mx.float32), best[:, None], xb], axis=1)
        total = mx.zeros((k, xb.shape[1] + 2), dtype=mx.float32).at[lab].add(vals).reshape(-1)
        return lab, best, total, mx.zeros_like(total)
    return step


def emulate_segments(x, best, perm, start, end, seg_first, k):
    """The segment and reduce kernels in NumPy float64, reading the layout exactly as they do."""
    n, d = x.shape
    w = d + 2
    ns = len(start)
    seg = np.zeros((ns, w))
    for s in range(ns):
        for p_ in range(start[s], end[s]):
            r = perm[p_]
            seg[s, 0] += 1
            seg[s, 1] += best[r]
            seg[s, 2:] += x[r]
    tot = np.zeros((k, w))
    for c in range(k):
        tot[c] = seg[seg_first[c]:seg_first[c + 1]].sum(0)
    return tot, seg


def mlx_cpu_part():
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    mx.set_memory_limit(2 << 30)
    print("== MLX on the CPU device: layout, two-sum, and the fit's plumbing with an MLX-op batch step")
    rng = np.random.default_rng(21)
    # segment layout: empty clusters, one cluster spanning several segments, clusters of exactly 256 and 257 rows
    k, n, d = 40, 3000, 5
    lab = rng.integers(0, 30, n)                                  # clusters 30..39 empty
    lab[:600] = 7                                                 # 600+ rows: three segments
    lab[600:856] = 31                                             # exactly 256
    lab[856:1113] = 33                                            # 257
    rng.shuffle(lab)
    x = rng.standard_normal((n, d)).astype(np.float32)
    best = rng.random(n).astype(np.float32)
    perm, start, end, seg_first, ns = em._segment_layout(mx.array(lab.astype(np.uint32)), n, k)
    perm, start, end, seg_first = (np.array(a).astype(np.int64) for a in (perm, start, end, seg_first))
    tot, seg = emulate_segments(x.astype(np.float64), best.astype(np.float64), perm, start, end, seg_first, k)
    cnt = np.bincount(lab, minlength=k)
    sums = np.zeros((k, d))
    np.add.at(sums, lab, x.astype(np.float64))
    covered = np.zeros(n, np.int64)
    one_cluster = True
    for s in range(ns):
        rows_s = perm[start[s]:end[s]]
        covered[rows_s] += 1
        one_cluster &= len(set(lab[rows_s])) <= 1 and end[s] - start[s] <= 256
    ok = (np.array_equal(tot[:, 0], cnt) and np.allclose(tot[:, 2:], sums, atol=1e-9) and np.all(covered == 1)
          and one_cluster and seg_first[-1] == int(np.sum(-(-cnt // 256))) and ns == -(-n // 256) + min(k, n)
          and np.allclose(tot[:, 1], np.bincount(lab, weights=best.astype(np.float64), minlength=k)))
    report(ok, "segment layout planned in MLX ops (emulated kernel)",
           f"{seg_first[-1]} segments used of {ns}, every row once, one cluster and <= 256 rows per segment")

    v = rng.standard_normal((400, 64)).astype(np.float32) * np.float32(1e3) + np.float32(1e4)
    s_, c_ = mx.zeros((64,)), mx.zeros((64,))
    for row in v:
        s_, c_ = em._two_sum_add(s_, c_, mx.array(row), mx.zeros((64,)))
    tot2 = np.array(s_, np.float64) + np.array(c_, np.float64)
    ref2 = v.astype(np.float64).sum(0)
    rel = float(np.max(np.abs(tot2 - ref2) / np.abs(ref2)))
    plain = float(np.max(np.abs(v.sum(0, dtype=np.float32) - ref2) / np.abs(ref2)))
    report(rel <= 1e-12, "two-sum accumulation in MLX ops", f"rel err {rel:.1e} (plain float32 sum {plain:.1e})")

    cases = [
        ("cpu: separated d8 k16, law (a0 = 1)", blobs(rng, 20_000, 8, 16, 10, 1.0), 16, {}, True),
        ("cpu: overlapping d16 k64, law", blobs(rng, 30_000, 16, 16, 3, 1.0), 64, {}, False),
        ("cpu: d12 k32, B 777, a0 0.3", blobs(rng, 12_000, 12, 8, 5, 1.0), 32, dict(batch_size=777, step0=0.3), False),
        # B = k/2 and 4 rows per cluster: hundreds revived per check, clusters left empty in the last epoch keep
        # their moving-average centre (the kept-centre branch of the final check)
        ("cpu: k2048, 4 rows per cluster, many revivals", blobs(rng, 8192, 8, 64, 5, 1.0), 2048,
         dict(batch_size=1024, step0=0.2), False),
    ]
    hook = ("test-hook", "test-hook")
    for name, X, kk, kw, free in cases:
        check_fit(name, X, kk, free=free, epochs=3, seed=3, _step=mlx_step(kk), expect=hook, **kw)
    # revival from far centres, over three slices of the data
    X = blobs(rng, 20_000, 16, 16, 10, 1.0)
    init = X[em._init_rows(9, len(X), 16)].copy()
    init[:3] = 1000.0
    _, rec, tr = check_fit("cpu: far centres revived, 3 slices", X, 16, per=7001, epochs=3, seed=9, init=init,
                           _step=mlx_step(16), expect=hook)
    report(sum(rec["revived"]) >= 3 and np.all(tr[0]["N"][:3] == 0) and np.array_equal(tr[0]["C"][:3], init[:3]),
           "cpu: empty clusters keep their centre at a0 = 1, then revive", f"revived per check {rec['revived'][:4]}...")
    # determinism, and independence from the slicing
    X = blobs(rng, 25_000, 16, 16, 3, 1.0)
    a, ra = em.emeans_fit(make_parts(X), 64, epochs=3, seed=17, _step=mlx_step(64))
    b, rb = em.emeans_fit(make_parts(X), 64, epochs=3, seed=17, _step=mlx_step(64))
    c, rc = em.emeans_fit(make_parts(X, 9_999), 64, epochs=3, seed=17, _step=mlx_step(64))
    dd, _ = em.emeans_fit(make_parts(X), 64, epochs=3, seed=18, _step=mlx_step(64))
    ok = (np.array_equal(a, b) and ra["train_mse"] == rb["train_mse"] and np.array_equal(a, c)
          and ra["train_mse"] == rc["train_mse"] and not np.array_equal(a, dd))
    report(ok, "cpu: same seed same bits; 1 vs 3 slices same bits", f"train MSE {ra['train_mse']}")
    _, rn = em.emeans_fit(make_parts(X), 64, epochs=2, seed=None, _step=mlx_step(64))
    report(isinstance(rn["seed"], int) and rn["seed"] > 0, "cpu: seed=None draws and records a seed")


# ---------------------------------------------------------------- GPU part

def wrong_vs_float64(X, C, lab, tol=1e-5):
    X64, C64 = np.asarray(X, np.float64), np.asarray(C, np.float64)
    csq = (C64 * C64).sum(1)
    bad = 0
    for s in range(0, len(X64), 20000):
        x = X64[s:s + 20000]
        d2 = (x * x).sum(1)[:, None] + csq[None] - 2 * x @ C64.T
        excess = d2[np.arange(len(x)), lab[s:s + 20000]] / np.maximum(d2.min(1), 1e-300) - 1
        bad += int(np.sum(excess > tol))
    return bad


def gpu_part():
    import gc
    import mlx.core as mx
    from mlx_kmeans import KMeans, core
    mx.set_default_device(mx.gpu)                                 # the MLX-CPU part ran on the CPU device
    mx.set_memory_limit(int(float(os.environ.get("EMEANS_TEST_MEM_GB", "8")) * (1 << 30)))
    print("== GPU: the fit against the reference, determinism, the estimator, caches")
    rng = np.random.default_rng(31)

    def fresh():
        core._approx_state.clear(); core._cascade_state.clear(); core._tiles1_slow.clear(); core._bounds_state.clear()

    fresh()
    X = blobs(rng, 20_000, 8, 16, 10, 1.0)
    for method, acc, want in [("auto", "auto", ("onehot", "fused")), ("rows", "lanes", ("rows", "lanes")),
                              ("rows", "blocks", ("rows", "blocks")), ("rows", "segments", ("rows", "segments"))]:
        check_fit(f"separated d8 k16, law (a0 = 1), {method}/{acc}", X, 16, free=True, epochs=3, seed=1,
                  method=method, accumulate=acc, expect=want)
    X = blobs(rng, 30_000, 48, 32, 3, 1.0)
    for method, acc, want in [("auto", "auto", ("onehot", "fused")), ("tiles1", "lanes", ("tiles1", "lanes")),
                              ("rows", "segments", ("rows", "segments"))]:
        check_fit(f"overlapping d48 k64, {method}/{acc}", X, 64, epochs=3, seed=2, method=method, accumulate=acc,
                  expect=want)
    X = blobs(rng, 100_000, 32, 64, 3, 1.0)
    check_fit("d32 k256 n=100k, a0 < 1 (tiles1 probe, segments)", X, 256, epochs=3, seed=3,
              expect=("tiles1", "segments", "rows"))
    X = blobs(rng, 40_000, 128, 64, 4, 1.0) * np.linspace(8.0, 0.05, 128).astype(np.float32)
    for method, want in [("auto", ("tiles1", "segments", "tiles")), ("tiles", ("tiles", "segments")),
                         ("pairs", ("pairs", "segments"))]:
        check_fit(f"anisotropic d128 k256, {method}", X, 256, epochs=2, seed=4, method=method, expect=want)
    X = blobs(rng, 30_000, 96, 16, 3, 1.0)
    for method, want in [("auto", ("onehot", "fused")), ("pairs", ("pairs", "segments"))]:
        check_fit(f"d96 k32, B 3000, {method}", X, 32, epochs=3, seed=5, batch_size=3000, method=method, expect=want)
    X = blobs(rng, 8_000, 960, 50, 1, 1.0)
    check_fit("GIST-like d960 k64 (tiles, segments)", X, 64, epochs=3, seed=6, expect=("tiles", "segments"))
    X = blobs(rng, 60_000, 128, 512, 4, 1.0)
    check_fit("d128 k4096 (tiles1 probe, segments at large k)", X, 4096, epochs=2, seed=7,
              expect=("tiles1", "segments", "tiles"))
    # uncentred data puts every centre inside tiles1's error window: the first-batch probe should see a high
    # fallback rate and move the fit to rows (d48 is below the tiles and pairs thresholds); judged for consistency
    # with the measured rate, not for which side of the limit it lands
    X = blobs(rng, 30_000, 48, 32, 3, 1.0) + np.float32(1000)
    _, rec, _ = check_fit("uncentred +1000 d48 k256 (tiles1 probe)", X, 256, epochs=3, seed=18,
                          expect=("tiles1", "segments", "rows"))
    X = blobs(rng, 20_003, 16, 16, 10, 1.0)
    init = X[em._init_rows(8, len(X), 100)].copy()
    init[:5] = 1000.0
    _, rec, _ = check_fit("far centres revived, 3 slices, ragged tail", X, 100, per=7001, epochs=3, seed=8, init=init,
                          expect=("rows", "segments"))
    report(sum(rec["revived"]) >= 5, "revival fired", f"{rec['revived'][:5]}...")
    X = blobs(rng, 30_000, 32, 32, 3, 1.0)
    check_fit("k-means++ init d32 k128", X, 128, epochs=3, seed=9, init="k-means++", expect=("onehot", "fused"))
    Xn = blobs(rng, 40_000, 256, 40, 3, 1.0)
    Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)                # normalised: float16 admitted
    fresh()
    _, rec, _ = check_fit("approx float16 d256 k64", Xn, 64, approx=True, epochs=3, seed=10, assign="approx",
                          expect=("approx", "segments"))
    report(rec["assign"] == "approx" and rec["approx_float16"], "approx path ran in float16",
           f"assign {rec['assign']}, float16 {rec['approx_float16']}, method {rec['method']}")
    Xs = rng.random((30_000, 256), dtype=np.float32) * 255          # SIFT-like range: float16 refused, float32 matmul
    fresh()
    _, rec, _ = check_fit("approx float32 d256 k64", Xs, 64, approx=True, epochs=2, seed=11, assign="approx",
                          expect=("approx", "segments"))
    report(rec["assign"] == "approx" and not rec["approx_float16"], "approx path ran in float32 (float16 refused)",
           f"assign {rec['assign']}, float16 {rec['approx_float16']}")

    # a GPU that refuses the one-hot kernel: the first batch falls back to rows + an accumulator, and the run is
    # then the same as one that asked for rows (batch 0 is recomputed with rows before anything uses it)
    X = blobs(rng, 20_000, 8, 16, 10, 1.0)
    want, _ = em.emeans_fit(make_parts(X), 16, epochs=2, seed=19, method="rows")
    orig, was_broken = core._onehot_totals, (8, 16) in core._onehot_broken

    def refuse(*a, **k_):
        raise RuntimeError("test: the one-hot kernel is refused")
    core._onehot_totals = refuse
    try:
        got, rec = em.emeans_fit(make_parts(X), 16, epochs=2, seed=19)
        broken = (8, 16) in core._onehot_broken
    finally:
        core._onehot_totals = orig
        if not was_broken:                         # leave a real refusal recorded; undo only the simulated one
            core._onehot_broken.discard((8, 16))
    report(rec["method"] == "rows" and rec["accumulate"] == em.choose_accumulator(rec["B"], 8, 16) and broken
           and np.array_equal(got, want), "one-hot refused: falls back to rows, same bits as method='rows'",
           f"{rec['method']}/{rec['accumulate']}, shape marked broken {broken}")

    # determinism and slicing, without the trace (the production loop)
    X = blobs(rng, 50_003, 128, 64, 4, 1.0)
    parts1, parts3 = make_parts(X), make_parts(X, 17_001)
    a, ra = em.emeans_fit(parts1, 256, epochs=3, seed=12)
    b, rb = em.emeans_fit(parts1, 256, epochs=3, seed=12)
    c, rc = em.emeans_fit(parts3, 256, epochs=3, seed=12)
    d_, _ = em.emeans_fit(parts1, 256, epochs=3, seed=13)
    ok = (np.array_equal(a, b) and ra["train_mse"] == rb["train_mse"] and np.array_equal(a, c)
          and ra["train_mse"] == rc["train_mse"] and not np.array_equal(a, d_))
    report(ok, "same seed same bits; 1 vs 3 slices same bits; other seed differs",
           f"{ra['method']}/{ra['accumulate']}, train {ra['timings']['train_s']:.3f}s")
    t_ = []
    e_, _ = em.emeans_fit(parts1, 256, epochs=3, seed=12, trace=t_)
    report(np.array_equal(a, e_), "the traced loop gives the same bits as the production loop")

    # KMeans(algorithm="emeans")
    fresh()
    X = blobs(rng, 60_000, 32, 32, 3, 1.0)
    m = KMeans(n_clusters=128, algorithm="emeans", epochs=3, random_state=0).fit(X)
    X64, C64 = X.astype(np.float64), m.cluster_centers_.astype(np.float64)
    ref_inertia = float(((X64 - C64[m.labels_]) ** 2).sum())
    bad = wrong_vs_float64(X, m.cluster_centers_, m.labels_)
    m2 = KMeans(n_clusters=128, algorithm="emeans", epochs=3, random_state=0).fit(X)
    m3 = KMeans(n_clusters=128, algorithm="emeans", epochs=3, random_state=0, n_init=3).fit(X)
    singles = [KMeans(n_clusters=128, algorithm="emeans", epochs=3, random_state=r).fit(X).inertia_ for r in (0, 1, 2)]
    lloyd = KMeans(n_clusters=128, random_state=0).fit(X)
    ok = (bad == 0 and abs(m.inertia_ - ref_inertia) <= 1e-6 * ref_inertia and m.n_iter_ == 3
          and m.emeans_record_["epochs"] == 3 and np.array_equal(m.cluster_centers_, m2.cluster_centers_)
          and m3.inertia_ == min(singles))
    report(ok, "KMeans(algorithm='emeans'): exact final pass, seeds, n_init",
           f"{bad} wrong labels, inertia rel err {abs(m.inertia_ - ref_inertia) / ref_inertia:.1e}, "
           f"e-means/Lloyd inertia {m.inertia_ / lloyd.inertia_:.4f} (Lloyd {lloyd.n_iter_} iterations)")
    B = (rng.standard_normal((20_000, 32)) * 3).astype(np.float32)
    report(wrong_vs_float64(B, m.cluster_centers_, m.predict(B)) == 0, "KMeans(algorithm='emeans').predict")
    me = KMeans(n_clusters=64, algorithm="emeans", exact=False, random_state=0).fit(Xn)
    bad = wrong_vs_float64(Xn, me.cluster_centers_, me.labels_)
    report(bad <= 0.01 * len(Xn), "KMeans(algorithm='emeans', exact=False)", f"{bad} of {len(Xn)} labels off by > 1e-5")
    raised = 0
    for kw in [dict(algorithm="minibatch"), dict(algorithm="emeans", epochs=0), dict(algorithm="emeans", spherical=True),
               dict(algorithm="emeans", accumulate="atomic"), dict(algorithm="emeans", batch_size=10 ** 9),
               dict(algorithm="emeans", batch_size=2.5)]:
        try:
            KMeans(n_clusters=8, **kw).fit(X[:5000])
        except ValueError:
            raised += 1
    report(raised == 6, "KMeans refuses bad e-means options", f"{raised}/6 raised ValueError")

    # per-dataset caches: e-means must not leave state keyed on batch arrays, nor evict the dataset's own
    caches = {"approx": core._approx_state, "cascade": core._cascade_state, "tiles1": core._tiles1_slow,
              "bounds": core._bounds_state}

    def snapshot():
        return {nm: {key: ent for key, ent in c.items()} for nm, c in caches.items()}

    def only_full(parts):
        return all(core._holds(ent, parts) for c in caches.values() for ent in c.values())

    fresh()
    X = blobs(rng, 60_000, 128, 64, 4, 1.0) * np.linspace(8.0, 0.05, 128).astype(np.float32)
    parts = make_parts(X)
    C = X[:256].copy()
    for _ in range(3):                                           # Lloyd builds the cascade state for this dataset
        C, _, _ = core.lloyd_step(parts, C)
    before = snapshot()
    em.emeans_fit(parts, 256, epochs=2, seed=14)
    after = snapshot()
    same = all(before[nm].keys() == after[nm].keys() and all(before[nm][key] is after[nm][key] for key in before[nm])
               for nm in caches)
    report(same and len(before["cascade"]) == 1 and only_full(parts), "exact e-means leaves the dataset's caches alone",
           f"{ {nm: len(c) for nm, c in after.items()} }")
    fresh()
    X = blobs(rng, 40_000, 512, 40, 3, 1.0)
    parts = make_parts(X)
    C = X[:64].copy()
    for _ in range(6):                                           # Lloyd's per-centre bounds for this dataset
        C, _, _ = core.lloyd_step(parts, C)
    before = snapshot()
    em.emeans_fit(parts, 64, epochs=2, seed=15)
    after = snapshot()
    same = all(before[nm].keys() == after[nm].keys() and all(before[nm][key] is after[nm][key] for key in before[nm])
               for nm in caches)
    report(same and len(before["bounds"]) == 1 and only_full(parts), "exact e-means keeps Lloyd's bounds state",
           f"{ {nm: len(c) for nm, c in after.items()} }")
    fresh()
    parts = make_parts(Xn)
    em.emeans_fit(parts, 64, epochs=2, seed=16, assign="approx")
    st = list(core._approx_state.values())
    half = st[0]["half"] if st else None
    em.emeans_fit(parts, 64, epochs=2, seed=17, assign="approx")
    st2 = list(core._approx_state.values())
    ok = (len(st) == 1 and core._holds(st[0], parts) and half is not None and len(st2) == 1 and st2[0] is st[0]
          and st2[0]["half"] is half and only_full(parts))
    report(ok, "approx e-means: one float16 copy of the full data, reused", f"{len(st2)} entry")
    del parts, st, st2, half
    gc.collect()
    report(not core._approx_state, "approx state released with the data", f"{len(core._approx_state)} entries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numpy", action="store_true", help="NumPy part only (imports no MLX)")
    ap.add_argument("--mlx-cpu", action="store_true", help="NumPy part, then MLX on the CPU device (no Metal kernels)")
    a = ap.parse_args()
    t = time.perf_counter()
    numpy_part()
    if a.numpy:
        assert "mlx" not in sys.modules and "mlx.core" not in sys.modules, "the NumPy part imported MLX"
    elif a.mlx_cpu:
        mlx_cpu_part()
    else:
        mlx_cpu_part()
        gpu_part()
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed in {time.perf_counter() - t:.0f} s")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
