#!/usr/bin/env python3
"""Benchmark k-means implementations head to head on this machine.

Every implementation gets the same data, the same k-means++ starting centers and the same number of full Lloyd
passes (no early stopping, no subsampling). An untimed warm-up handles transfers, kernel compilation and thread
pools; then the pass loop is timed REPEATS times and the median is reported.

Trust checks per result:
  - inertia of the final centers is recomputed the same way for every implementation and compared with ours
    (ours is verified against float64 by tests/accuracy.py); > 1e-4 relative difference = "different work",
    excluded from the fastest-public pick.
  - if ours ever sees an empty cluster, libraries legitimately diverge (each re-seeds differently): the config
    is flagged as not comparable.
  - 1-minute load average is recorded before each timing; > 4 is flagged.

Each run appends one JSON line per implementation to benchmarks/results.jsonl; BENCHMARKS.md is regenerated.

  .venv/bin/python bench.py --suite                 # representative suite (see SUITE)
  .venv/bin/python bench.py --config sift-k1024     # one suite config
  .venv/bin/python bench.py --report                # only rebuild BENCHMARKS.md
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
PUBLISH_TARGET = 1.5
ITERS, REPEATS = 5, 3

# Representative problems (see google_chat.txt): shapes people actually cluster.
SUITE = {
    "geo-trips":      dict(data="synthetic", rows=10_000_000, dims=4, k=256,
                           why="GPS / origin-destination points"),
    "satellite":      dict(data="synthetic", rows=10_000_000, dims=12, k=32,
                           why="multispectral pixels (Sentinel-2 has 13 bands)"),
    "logs":           dict(data="synthetic", rows=10_000_000, dims=32, k=256,
                           why="network/security feature vectors"),
    "single-cell":    dict(data="synthetic", rows=2_000_000, dims=50, k=64,
                           why="scRNA-seq after PCA to 50 components"),
    "sift-k1024":     dict(data="sift", rows=1_000_000, dims=128, k=1024,
                           why="SIFT1M: vector-search training (IVF coarse quantizer)"),
    "gist-k1024":     dict(data="gist", rows=1_000_000, dims=960, k=1024,
                           why="GIST1M: high-dimensional vector-search training"),
}


# ---------------------------------------------------------------- implementations
# Each takes (X numpy float32, parts mlx list, C0) and returns run(iters) -> final centers.

def ours_metal(X, parts, C0):
    def run(iters):
        C = C0.copy()
        empty = False
        for _ in range(iters):
            sums, counts, _ = km.assign_mlx(parts, C)
            has = counts > 0
            empty |= not has.all()
            C[has] = (sums[has] / counts[has, None]).astype(np.float32)
        run.saw_empty = empty
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
# Memory guards. GPU memory on Apple Silicon is wired RAM: an oversized MPS allocation can't be paged out and can
# freeze the whole machine (2026-09-17: fast-pytorch-kmeans on GIST1M wired 123 GB -> watchdog kernel panic).
# fast-pytorch-kmeans sizes its chunks by test-allocating up to rows * dims * k * 4 bytes on the GPU and retrying
# smaller, so PyTorch's MPS allocator is capped (verified: oversized allocations then fail cleanly and the probe
# retries smaller). Must be set before torch is imported; the low watermark must not exceed the high one.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.3")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.25")
# Steady-state extra memory beyond the data (dense k x rows mask + a GPU copy of the data), to skip runs that swap.
MEMORY_GB = {"fast-pytorch-kmeans (MPS GPU)": lambda rows, dims, k: rows * k * 4 / 1e9 + rows * dims * 4 / 1e9}
MAX_EXTRA_GB = 24
# scikit-learn runs one extra assignment pass after max_iter, so its time is divided by iters + 1 (in its favour).
EXTRA_PASSES = {"scikit-learn lloyd (CPU)": 1, "scikit-learn elkan (CPU)": 1}


# ---------------------------------------------------------------- data

def read_fvecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    return raw.reshape(-1, d + 1)[:, 1:].view(np.float32).copy()


def load(cfg, seed):
    mx = km._mx()
    if cfg["data"] == "synthetic":
        parts = km.make_data_mlx(cfg["rows"], cfg["dims"], cfg["k"], seed)
        X = np.concatenate([np.array(p) for p in parts])
    else:
        X = read_fvecs(ROOT / "data" / cfg["data"] / f"{cfg['data']}_base.fvecs")[:cfg["rows"]]
        assert X.shape[1] == cfg["dims"], X.shape
        per = km.slice_rows(X.shape[1])
        parts = [mx.array(X[s:s + per]) for s in range(0, len(X), per)]
    mx.eval(parts)
    return X, parts


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


def bench(name, cfg, only, seed=0):
    X, parts = load(cfg, seed)
    rows, dims, k = X.shape[0], X.shape[1], cfg["k"]
    C0 = km.kmeans_pp_init(parts, rows, k, np.random.default_rng(seed))
    commit, dirty = git_state()
    results, ours_inertia, ours_empty = [], None, False
    print(f"== {name}: {rows:,} rows, {dims} dims, k={k} ({cfg['data']})")
    for impl, (make, module) in IMPLS.items():
        if only and impl != OURS and not any(o.lower() in impl.lower() for o in only):
            continue
        print(f"   {impl:32s} ", end="", flush=True)
        rec = dict(date=datetime.datetime.now().isoformat(timespec="seconds"), commit=commit, dirty=dirty,
                   machine=machine(), config=name, data=cfg["data"], impl=impl, version=version(module),
                   rows=rows, dims=dims, k=k, iters=ITERS, repeats=REPEATS)
        need = MEMORY_GB.get(impl, lambda *a: 0)(rows, dims, k)
        if need > MAX_EXTRA_GB:
            rec.update(error=f"skipped: needs ~{need:.0f} GB extra memory (limit {MAX_EXTRA_GB} GB)")
            print(rec["error"])
            results.append(rec)
            continue
        try:
            run = make(X, parts, C0)
            run(1)  # warm-up
            times, loads = [], []
            for _ in range(REPEATS):
                loads.append(os.getloadavg()[0])
                t = time.perf_counter()
                C = run(ITERS)
                times.append((time.perf_counter() - t) / (ITERS + EXTRA_PASSES.get(impl, 0)))
            inertia = km.assign_mlx(parts, np.ascontiguousarray(C, dtype=np.float32))[2]
            rec.update(sec_per_pass=float(np.median(times)), min_s=min(times), max_s=max(times),
                       load1=max(loads), inertia=inertia)
            if impl == OURS:
                ours_inertia, ours_empty = inertia, run.saw_empty
                rec["saw_empty_cluster"] = ours_empty
            else:
                rec["inertia_rel_vs_ours"] = (inertia - ours_inertia) / ours_inertia
                rec["valid"] = abs(rec["inertia_rel_vs_ours"]) <= 1e-4
            flags = ("  LOAD>4" if rec["load1"] > 4 else "") + ("" if rec.get("valid", True) else "  DIFFERENT WORK")
            print(f"{rec['sec_per_pass']:.4f} s/pass (min {rec['min_s']:.4f}, max {rec['max_s']:.4f})"
                  f"  inertia {inertia:.6e}{flags}")
        except Exception as e:
            rec.update(error=f"{type(e).__name__}: {str(e)[:200]}")
            print("FAILED", rec["error"])
        results.append(rec)
    for r in results:
        r["config_comparable"] = not ours_empty
    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    del X, parts
    km._mx().clear_cache()


def report():
    rows = [json.loads(line) for line in RESULTS.read_text().splitlines()] if RESULTS.exists() else []
    rows = [r for r in rows if r.get("config") in SUITE]  # suite results only; older ad-hoc runs stay in the jsonl
    latest = {}
    for r in rows:
        latest[(r["config"], r["impl"])] = r
    out = [
        "# Benchmarks",
        "",
        f"Generated by `bench.py` from `benchmarks/results.jsonl` (full history). Machine: {rows[-1]['machine'] if rows else '?'}.",
        "",
        f"**Method:** identical data, identical k-means++ starting centers, {ITERS} full Lloyd passes per timed run "
        f"(no early stopping; FAISS's 256-points-per-cluster subsampling disabled), untimed warm-up, median of "
        f"{REPEATS} timed runs. scikit-learn's extra final assignment pass is counted in its favour.",
        "",
        "**Trust checks:** our pass is verified against float64 by `tests/accuracy.py` (every label, centers, inertia). "
        "A library whose final inertia differs from ours by more than 1e-4 relative did different work and is excluded "
        "from the comparison. A config where our run hit an empty cluster is flagged not comparable. Load average > 4 "
        "during timing is flagged.",
        "",
        f"**Publish target:** ours at least {PUBLISH_TARGET}x faster than the fastest valid public implementation on every config.",
        "",
        "| Config | Problem | Shape | Ours s/pass | Fastest valid public | Its s/pass | Ours vs it |",
        "|---|---|---|---|---|---|---|",
    ]
    details = []
    for name, cfg in SUITE.items():
        rs = [latest[(name, i)] for i in IMPLS if (name, i) in latest]
        if not rs:
            continue
        ours = next((r for r in rs if r["impl"] == OURS and "sec_per_pass" in r), None)
        public = [r for r in rs if r["impl"] != OURS and "sec_per_pass" in r and r.get("valid")]
        best = min(public, key=lambda r: r["sec_per_pass"]) if public else None
        shape = f"{rs[0]['rows']:,} x {rs[0]['dims']}, k={rs[0]['k']}"
        comparable = all(r.get("config_comparable", True) for r in rs)
        if ours and best:
            ratio = best["sec_per_pass"] / ours["sec_per_pass"]
            mark = "✅" if ratio >= PUBLISH_TARGET else ("🟡" if ratio >= 1 else "❌")
            verdict = f"{mark} {ratio:.1f}x" + ("" if comparable else " (not comparable: empty cluster)")
            out.append(f"| {name} | {cfg['why']} | {shape} | {ours['sec_per_pass']:.4f} | {best['impl']} | {best['sec_per_pass']:.4f} | {verdict} |")
        details += [f"### {name}: {cfg['why']} ({shape}, {cfg['data']} data)", "",
                    "| Implementation | Version | s/pass (median) | min–max | vs ours | Inertia vs ours | Load | Commit | Date |",
                    "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(rs, key=lambda r: r.get("sec_per_pass", float("inf"))):
            if "sec_per_pass" not in r:
                details.append(f"| {r['impl']} | {r['version']} | {r['error'][:70]} | | | | | {r['commit']} | {r['date'][:10]} |")
                continue
            is_ours = r["impl"] == OURS
            vs = "—" if is_ours or not ours else f"{r['sec_per_pass'] / ours['sec_per_pass']:.1f}x slower" \
                if r["sec_per_pass"] >= ours["sec_per_pass"] else f"{ours['sec_per_pass'] / r['sec_per_pass']:.1f}x faster"
            rel = "—" if is_ours else f"{r['inertia_rel_vs_ours']:+.1e}" + ("" if r.get("valid") else " ⚠️ different work")
            commit = f"{r['commit']}{'*' if r['dirty'] else ''}"
            details.append(f"| {r['impl']} | {r['version']} | {r['sec_per_pass']:.4f} | {r['min_s']:.4f}–{r['max_s']:.4f} | {vs} | "
                           f"{rel} | {r['load1']:.1f} | {commit} | {r['date'][:10]} |")
        details.append("")
    out += ["", "✅ meets target · 🟡 faster but below target · ❌ slower", "", "## Details", ""] + details
    out += ["`*` after a commit = uncommitted changes to kmeans.py/bench.py at run time.", ""]
    REPORT.write_text("\n".join(out))
    print(f"wrote {REPORT.relative_to(ROOT)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", action="store_true", help="run every config in SUITE")
    p.add_argument("--config", nargs="*", default=[], choices=list(SUITE), help="run these suite configs")
    p.add_argument("--only", nargs="*", help="substring filter on public implementation names (ours always runs)")
    p.add_argument("--report", action="store_true", help="only regenerate BENCHMARKS.md")
    a = p.parse_args()
    names = list(SUITE) if a.suite else a.config
    if not a.report:
        for name in names:
            bench(name, SUITE[name], a.only)
    report()


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
