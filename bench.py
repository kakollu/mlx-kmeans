#!/usr/bin/env python3
"""Benchmark k-means implementations head to head on this machine.

Every implementation gets the same data, the same starting centers and the same number of full Lloyd passes
(no early stopping, no subsampling). Data transfer and kernel compilation happen in an untimed warm-up.
Each run appends one JSON line per implementation to benchmarks/results.jsonl, and BENCHMARKS.md is
regenerated from that file.

Run with the venv that has the public baselines (faiss-cpu, torch, fast-pytorch-kmeans):
  .venv/bin/python bench.py --rows 10_000_000 --dims 8 --k 16
  .venv/bin/python bench.py --report          # only rebuild BENCHMARKS.md
"""
import argparse
import datetime
import importlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

import kmeans as km

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "benchmarks" / "results.jsonl"
REPORT = ROOT / "BENCHMARKS.md"
OURS = "ours: kmeans.py metal"
PUBLISH_TARGET = 1.5  # publish if ours is >= 1.5x the fastest public implementation


# ---------------------------------------------------------------- implementations
# Each takes (X numpy float32, parts mlx list, C0, iters) and returns a runner: runner(iters) -> final centers.

def ours_metal(X, parts, C0):
    def run(iters):
        C = C0.copy()
        for _ in range(iters):
            sums, counts, _ = km.assign_mlx(parts, C)
            has = counts > 0
            C[has] = (sums[has] / counts[has, None]).astype(np.float32)
        return C
    return run


def sklearn_impl(algorithm):
    def make(X, parts, C0):
        from sklearn.cluster import KMeans

        def run(iters):
            m = KMeans(n_clusters=len(C0), init=C0, n_init=1, max_iter=iters, tol=0.0, algorithm=algorithm)
            m.fit(X)
            return m.cluster_centers_.astype(np.float32)
        return run
    return make


def faiss_cpu(X, parts, C0):
    import faiss

    def run(iters):
        # max_points_per_centroid: FAISS otherwise trains on a 256-points-per-cluster subsample
        m = faiss.Kmeans(X.shape[1], len(C0), niter=iters, max_points_per_centroid=2**31 - 1, verbose=False)
        m.train(X, init_centroids=C0)
        return m.centroids.copy()
    return run


def fast_pytorch_kmeans_mps(X, parts, C0):
    import torch
    from fast_pytorch_kmeans import KMeans
    Xt, Ct = torch.from_numpy(X).to("mps"), torch.from_numpy(C0).to("mps")

    def run(iters):
        m = KMeans(n_clusters=len(C0), max_iter=iters, tol=-1.0)
        m.fit_predict(Xt, centroids=Ct.clone())
        torch.mps.synchronize()
        return m.centroids.cpu().numpy()
    return run


IMPLS = {
    OURS: (ours_metal, "mlx.core"),
    "scikit-learn lloyd (CPU)": (sklearn_impl("lloyd"), "sklearn"),
    "scikit-learn elkan (CPU)": (sklearn_impl("elkan"), "sklearn"),
    "faiss (CPU)": (faiss_cpu, "faiss"),
    "fast-pytorch-kmeans (MPS GPU)": (fast_pytorch_kmeans_mps, "fast_pytorch_kmeans"),
}
# Rough extra memory beyond the data, to skip runs that would swap this shared machine.
MEMORY_GB = {"fast-pytorch-kmeans (MPS GPU)": lambda rows, dims, k: rows * k * 4 / 1e9 + rows * dims * 4 / 1e9}  # dense k x rows mask
MAX_EXTRA_GB = 24

# scikit-learn runs one extra assignment pass after max_iter (to make labels match centers), so its
# time is divided by iters + 1 -- in its favour.
EXTRA_PASSES = {"scikit-learn lloyd (CPU)": 1, "scikit-learn elkan (CPU)": 1}


# ---------------------------------------------------------------- run + record

def version(module):
    try:
        if module == "fast_pytorch_kmeans":
            from importlib.metadata import version as v
            return v("fast-pytorch-kmeans")
        return importlib.import_module(module).__version__
    except Exception:
        return None


def git_state():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "kmeans.py", "bench.py"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip())
        return commit, dirty
    except Exception:
        return None, None


def machine():
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) // 2**30
    return f"{chip}, {mem} GB, macOS {platform.mac_ver()[0]}"


def bench(rows, dims, k, iters, only, seed):
    import mlx.core as mx
    parts = km.make_data_mlx(rows, dims, k, seed)
    mx.eval(parts)
    X = np.concatenate([np.array(p) for p in parts])
    C0 = km.kmeans_pp_init(parts, rows, k, np.random.default_rng(seed))
    commit, dirty = git_state()
    results = []
    for name, (make, module) in IMPLS.items():
        if only and not any(o.lower() in name.lower() for o in only):
            continue
        print(f"{name:32s} ", end="", flush=True)
        rec = dict(date=datetime.datetime.now().isoformat(timespec="seconds"), commit=commit, dirty=dirty,
                   machine=machine(), impl=name, version=version(module), rows=rows, dims=dims, k=k, iters=iters)
        need = MEMORY_GB.get(name, lambda *a: 0)(rows, dims, k)
        if need > MAX_EXTRA_GB:
            rec.update(error=f"skipped: needs ~{need:.0f} GB extra memory (limit {MAX_EXTRA_GB} GB)")
            print(rec["error"])
            results.append(rec)
            continue
        try:
            run = make(X, parts, C0)
            run(1)  # warm-up: transfers, kernel compilation, thread pools
            t = time.perf_counter()
            C = run(iters)
            total = time.perf_counter() - t
            rec.update(total_s=round(total, 4), sec_per_pass=round(total / (iters + EXTRA_PASSES.get(name, 0)), 5),
                       inertia=km.assign_mlx(parts, np.ascontiguousarray(C, dtype=np.float32))[2])
            print(f"{rec['sec_per_pass']:.4f} s/pass   inertia {rec['inertia']:.6e}")
        except Exception as e:
            rec.update(error=f"{type(e).__name__}: {str(e)[:200]}")
            print("FAILED", rec["error"])
        results.append(rec)
    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return results


def report():
    rows = [json.loads(line) for line in RESULTS.read_text().splitlines()] if RESULTS.exists() else []
    latest = {}  # newest result per (config, impl)
    for r in rows:
        latest[(r["rows"], r["dims"], r["k"], r["impl"])] = r
    configs = sorted({key[:3] for key in latest})
    out = [
        "# Benchmarks",
        "",
        f"Generated by `bench.py` from `benchmarks/results.jsonl` (full history). Machine: {rows[-1]['machine'] if rows else '?'}.",
        "",
        "**Method:** identical data, identical k-means++ starting centers and a fixed number of full Lloyd passes for "
        "every implementation (no early stopping, no subsampling: FAISS's 256-points-per-cluster sampling is disabled). "
        "Warm-up run first. Time is per pass (assignment + center update); scikit-learn's extra final assignment pass "
        "is counted in its favour. Final inertia is recomputed the same way for all, to check they did the same work.",
        "",
        f"**Publish target:** ours at least {PUBLISH_TARGET}x faster than the fastest public implementation on this machine.",
        "",
    ]
    for rows_, dims, k in configs:
        rs = [latest[(rows_, dims, k, i)] for i in IMPLS if (rows_, dims, k, i) in latest]
        ok = [r for r in rs if "sec_per_pass" in r]
        public = [r for r in ok if r["impl"] != OURS]
        ours = next((r for r in ok if r["impl"] == OURS), None)
        best_pub = min(public, key=lambda r: r["sec_per_pass"]) if public else None
        out += [f"## {rows_:,} rows, {dims} dims, k={k}", ""]
        if ours and best_pub:
            ratio = best_pub["sec_per_pass"] / ours["sec_per_pass"]
            verdict = "meets" if ratio >= PUBLISH_TARGET else "below"
            out += [f"Ours vs fastest public ({best_pub['impl']}): **{ratio:.1f}x** ({verdict} the {PUBLISH_TARGET}x target)", ""]
        out += ["| Implementation | Version | s/pass | vs ours | Inertia rel. to ours | Commit | Date |",
                "|---|---|---|---|---|---|---|"]
        for r in sorted(rs, key=lambda r: r.get("sec_per_pass", float("inf"))):
            if "sec_per_pass" not in r:
                out.append(f"| {r['impl']} | {r['version']} | failed: {r['error'][:60]} | | | {r['commit']} | {r['date'][:10]} |")
                continue
            vs = f"{r['sec_per_pass'] / ours['sec_per_pass']:.1f}x slower" if ours and r is not ours else "—"
            rel = f"{(r['inertia'] - ours['inertia']) / ours['inertia']:+.1e}" if ours and r is not ours else "—"
            commit = f"{r['commit']}{'*' if r['dirty'] else ''}"
            out.append(f"| {r['impl']} | {r['version']} | {r['sec_per_pass']:.4f} | {vs} | {rel} | {commit} | {r['date'][:10]} |")
        out.append("")
    out += ["`*` after a commit = uncommitted changes to kmeans.py/bench.py at run time.", ""]
    REPORT.write_text("\n".join(out))
    print(f"wrote {REPORT.relative_to(ROOT)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=10_000_000)
    p.add_argument("--dims", type=int, default=8)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", nargs="*", help="substring filter on implementation names")
    p.add_argument("--report", action="store_true", help="only regenerate BENCHMARKS.md")
    a = p.parse_args()
    if not a.report:
        print(f"rows={a.rows:,} dims={a.dims} k={a.k} iters={a.iters}")
        bench(a.rows, a.dims, a.k, a.iters, a.only, a.seed)
    report()


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
