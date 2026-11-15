#!/usr/bin/env python3
"""e-means against exact Lloyd and FAISS's Lloyd: loss vs epochs, wall time, held-out error, IVF recall.

bench_ann.py's framing: wall time to a finished set of centres, then the full-data inertia of those centres by the
library's exact pass (the same metric for every method), plus the held-out MSE on the dataset's query vectors and
recall@10 through a FAISS IVF index built from the centres. Adds what e-means needs: several epoch counts, several
seeds, and Lloyd's per-iteration curve, so the time Lloyd takes to reach each e-means loss can be read off.

Run (data from scripts/get_data.py sift / gist; MLX memory capped by --mem-gb):
    .venv/bin/python emeans_bench.py --data sift --k 1024
Options: --epochs 1 2 3 5 10   --seeds 0 1 2   --only emeans lloyd faiss sklearn   --no-recall   --faiss-max-k 16384
         --sklearn-max-k 4096   --mem-gb 32
Appends one JSON line per run to benchmarks/emeans_results.jsonl (the 2026-11-14 rows are the ones benchmarks/NOTES.md
and docs/e-means.md cite as L1-L61).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import mlx_kmeans as km  # noqa: E402
from mlx_kmeans import core, emeans  # noqa: E402
from bench import machine, other_cpu_percent, read_fvecs, git_state  # noqa: E402
from bench_ann import read_ivecs, ivf_recall, faiss_kmeans, inertia_full, sklearn_minibatch, NPROBE  # noqa: E402

OUT = ROOT / "benchmarks" / "emeans_results.jsonl"


def fresh():
    """Every timed run starts with no per-dataset state (kernels stay compiled): no method inherits another's."""
    core._approx_state.clear(); core._cascade_state.clear(); core._tiles1_slow.clear(); core._bounds_state.clear()


def power():
    try:
        return subprocess.run(["pmset", "-g", "ps"], capture_output=True, text=True).stdout.split("'")[1]
    except Exception:
        return "unknown"


def lloyd_curve(parts, rows, k, C, max_iter, tol, snap=25):
    """Exact Lloyd from C, as bench_ann's 'ours' loop -> (final C, iterations, curve, C at `snap` iterations, t at snap).

    curve[j] = (seconds at the end of step j + 1, inertia of the centres after j steps), as lloyd_step reports it
    (each step scores the centres it started from); the centres after j steps were in hand at curve[j - 1][0].
    """
    t0 = time.perf_counter()
    curve, prev, snapC, snapT = [], None, None, None
    for it in range(max_iter):
        newC, inertia, _ = core.lloyd_step(parts, C)
        curve.append((time.perf_counter() - t0, float(inertia)))
        C = newC
        if it + 1 == snap:
            snapC, snapT = C.copy(), time.perf_counter() - t0
        if tol > 0 and prev is not None and abs(prev - inertia) <= tol * prev:
            break
        prev = inertia
    return C, it + 1, curve, snapC, snapT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="sift", help="sift or gist")
    p.add_argument("--k", type=int, default=1024)
    p.add_argument("--epochs", type=int, nargs="*", default=[1, 2, 3, 5, 10])
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--only", nargs="*", default=["emeans", "lloyd", "faiss", "sklearn"])
    p.add_argument("--no-recall", action="store_true")
    p.add_argument("--faiss-max-k", type=int, default=16384, help="skip FAISS above this k (CPU time)")
    p.add_argument("--sklearn-max-k", type=int, default=4096, help="skip scikit-learn above this k (CPU time)")
    p.add_argument("--lloyd-max-iter", type=int, default=300)
    p.add_argument("--mem-gb", type=float, default=32, help="mx.set_memory_limit for this process (GB)")
    a = p.parse_args()
    import mlx.core as mx
    mx.set_memory_limit(int(a.mem_gb * 2**30))
    mx = km._mx()
    base = read_fvecs(ROOT / f"data/{a.data}/{a.data}_base.fvecs")
    queries = read_fvecs(ROOT / f"data/{a.data}/{a.data}_query.fvecs")
    gt = read_ivecs(ROOT / f"data/{a.data}/{a.data}_groundtruth.ivecs")
    parts = [mx.array(base)]
    qparts = [mx.array(queries)]
    mx.eval(parts, qparts)
    rows, k = len(base), a.k
    commit, dirty = git_state()
    print(f"{a.data}: {rows:,} x {base.shape[1]}, k={k}, {len(queries):,} queries; power {power()}; "
          f"law: {emeans.plan(rows, k, 3)} at T=3\n", flush=True)

    def score(C):
        C = np.ascontiguousarray(C, dtype=np.float32)
        fresh()
        inertia = inertia_full(parts, C)
        q_mse = km.assign_mlx(qparts, C)[2] / len(queries)
        fresh()
        return inertia, q_mse

    out = []

    def emit(rec):
        rec.update(date=time.strftime("%Y-%m-%dT%H:%M:%S"), commit=commit, dirty=dirty, machine=machine(),
                   power=power(), data=a.data, rows=rows, k=k)
        if not a.no_recall and rec.get("recall_this", True):
            r, build = ivf_recall(base, queries[:10_000], gt[:10_000], rec.pop("C"), k)
            rec["recall"] = {str(n_): r[n_][0] for n_ in NPROBE}
            rec["index_build_s"] = build
        rec.pop("C", None)
        rec.pop("recall_this", None)
        out.append(rec)
        with OUT.open("a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        rc = f"  recall@10 nprobe 8 {rec['recall']['8'] * 100:5.2f}%" if "recall" in rec else ""
        print(f"{rec['method']:44s} {rec['train_s']:8.3f}s  inertia {rec['inertia']:.6e}  query MSE {rec['query_mse']:.6e}"
              f"{rc}", flush=True)

    # warm-up: compile every kernel the timed runs use, untimed
    fresh()
    emeans.emeans_fit(parts, k, epochs=1, seed=99)
    core.lloyd_step(parts, core.take_rows(parts, np.arange(k)))
    fresh()

    if "emeans" in a.only:
        for T in a.epochs:
            fresh()
            emeans.emeans_fit(parts, k, epochs=T, seed=99)       # untimed: B follows T, and some kernels template on it
            for seed in a.seeds:
                fresh()
                load = other_cpu_percent()
                t = time.perf_counter()
                C, rec = emeans.emeans_fit(parts, k, epochs=T, seed=seed)
                train_s = time.perf_counter() - t
                inertia, q = score(C)
                emit(dict(method=f"e-means T={T}", epochs=T, seed=seed, train_s=train_s, inertia=inertia, query_mse=q,
                          other_cpu=load, C=C, recall_this=seed == a.seeds[0],
                          emeans={kk: v for kk, v in rec.items() if kk not in ("init_centres",)}))

    if "lloyd" in a.only:
        for init in ("k-means++", "random"):
            fresh()
            load = other_cpu_percent()
            t = time.perf_counter()
            if init == "k-means++":
                C0 = core.kmeans_pp_init(parts, rows, k, np.random.default_rng(0))
            else:
                C0 = core.take_rows(parts, emeans._init_rows(0, rows, k))     # e-means' starting rows, seed 0
            init_s = time.perf_counter() - t
            C, iters, curve, snapC, snapT = lloyd_curve(parts, rows, k, C0, a.lloyd_max_iter, 1e-4)
            train_s = init_s + curve[-1][0]
            inertia, q = score(C)
            emit(dict(method=f"Lloyd ({init}), to convergence", iters=iters, train_s=train_s, init_s=init_s,
                      inertia=inertia, query_mse=q, other_cpu=load, C=C, curve=curve))
            if snapC is not None:
                inertia, q = score(snapC)
                emit(dict(method=f"Lloyd ({init}), 25 iterations", iters=25, train_s=init_s + snapT, init_s=init_s,
                          inertia=inertia, query_mse=q, other_cpu=load, C=snapC))

    if "faiss" in a.only and k <= a.faiss_max_k:
        for sub in (True, False):
            load = other_cpu_percent()
            t = time.perf_counter()
            C, iters = faiss_kmeans(base, k, 0, subsample=sub)
            train_s = time.perf_counter() - t
            inertia, q = score(C)
            emit(dict(method=f"FAISS Lloyd {'default (subsample)' if sub else 'all rows'}, 25 iters", iters=iters,
                      train_s=train_s, inertia=inertia, query_mse=q, other_cpu=load, C=C))

    if "sklearn" in a.only and k <= a.sklearn_max_k:
        # The standard library's mini-batch k-means (Sculley), the paper's mini-batch baseline. One setting,
        # bench_ann's (batch 10,000, up to 100 iterations, its own early stop), as in the repo's 2026-09-18 record;
        # the paper swept batch size and reassignment ratio on a validation split, which is not repeated here.
        load = other_cpu_percent()
        t = time.perf_counter()
        C, iters = sklearn_minibatch(base, k, 0)
        train_s = time.perf_counter() - t
        inertia, q = score(C)
        emit(dict(method="scikit-learn MiniBatchKMeans (batch 10,000)", iters=iters, train_s=train_s, inertia=inertia,
                  query_mse=q, other_cpu=load, C=C))

    # summary: every run against the lowest inertia of Lloyd to convergence, and Lloyd's time to reach each e-means loss
    ref = min((r for r in out if r["method"].startswith("Lloyd") and "convergence" in r["method"]),
              key=lambda r: r["inertia"], default=None)
    if ref is None:
        return
    print(f"\nagainst {ref['method']} ({ref['inertia']:.6e}, {ref['train_s']:.2f}s):")
    curves = [r for r in out if "curve" in r]
    for r in out:
        line = f"  {r['method']:44s} seed {r.get('seed', '-')!s:>2}  inertia {r['inertia'] / ref['inertia']:.5f}x  " \
               f"query MSE {r['query_mse'] / ref['query_mse']:.5f}x  {r['train_s']:8.3f}s"
        if r["method"].startswith("e-means"):
            for c in curves:                      # first Lloyd iterate at or below this loss, and when it was in hand
                hit = next((j for j in range(1, len(c["curve"])) if c["curve"][j][1] <= r["inertia"]), None)
                line += (f" | {c['method'].split(',')[0]}: reaches it after {hit} it, {c['init_s'] + c['curve'][hit - 1][0]:.2f}s"
                         if hit else f" | {c['method'].split(',')[0]}: not reached")
        print(line)


if __name__ == "__main__":
    main()
