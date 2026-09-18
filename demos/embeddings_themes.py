#!/usr/bin/env python3
"""Demo: cluster 1M OpenAI embeddings of Wikipedia/DBpedia entities on a laptop.

Two things a vector database needs, both k-means:
  themes  - a small k, to see what a corpus is actually about (with the article titles nearest each centre)
  index   - a large k (an IVF coarse quantiser), which is how vector search avoids scanning every vector

Writes demos/embeddings_results.json.

  python3 demos/embeddings_themes.py [--themes 20] [--index-k 1024] [--compare]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx_kmeans as km  # noqa: E402
from mlx_kmeans import KMeans  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dbpedia"


def load(limit=None):
    """Embeddings (float32) and titles from the parquet shards."""
    t = time.perf_counter()
    vecs, titles = [], []
    for path in sorted(DATA.glob("shard*.parquet")):
        tab = pq.read_table(path, columns=["title", "openai"])
        v = np.stack(tab["openai"].to_numpy(zero_copy_only=False)).astype(np.float32)
        vecs.append(v)
        titles.extend(tab["title"].to_pylist())
        if limit and sum(len(x) for x in vecs) >= limit:
            break
    X = np.ascontiguousarray(np.concatenate(vecs)[:limit] if limit else np.concatenate(vecs))
    titles = titles[:len(X)]
    print(f"{len(X):,} embeddings x {X.shape[1]} dims ({X.nbytes/1e9:.1f} GB) read in {time.perf_counter()-t:.1f}s")
    return X, titles, time.perf_counter() - t


def nearest_titles(parts, X, centers, titles, per=6):
    """Titles closest to each centre, so a theme can be read rather than guessed."""
    labels = km.assign_mlx(parts, np.ascontiguousarray(centers, dtype=np.float32), return_labels=True)[3]
    keep = [[] for _ in centers]
    for c in range(len(centers)):
        idx = np.flatnonzero(labels == c)[:200_000]        # a centre's own members, capped for the distance sort
        if not len(idx):
            continue
        d = ((X[idx] - centers[c]) ** 2).sum(1)
        keep[c] = [titles[int(idx[j])] for j in np.argsort(d)[:per]]
    return keep, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--themes", type=int, default=20)
    p.add_argument("--index-k", type=int, default=1024)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--compare", action="store_true")
    a = p.parse_args()

    X, titles, read_s = load(a.limit)
    mx = km._mx()
    per = km.slice_rows(X.shape[1])
    parts = [mx.array(X[s:s + per]) for s in range(0, len(X), per)]
    mx.eval(parts)

    t = time.perf_counter()
    themes = KMeans(n_clusters=a.themes, random_state=0).fit(parts)
    theme_s = time.perf_counter() - t
    print(f"themes: {a.themes} clusters over {len(X):,} x {X.shape[1]} in {theme_s:.1f}s ({themes.n_iter_} iterations)")

    t = time.perf_counter()
    index = KMeans(n_clusters=a.index_k, random_state=0).fit(parts)
    index_s = time.perf_counter() - t
    print(f"index:  {a.index_k} clusters (IVF quantiser) in {index_s:.1f}s ({index.n_iter_} iterations)")

    titles_by_theme, labels = nearest_titles(parts, X, themes.cluster_centers_, titles)
    counts = np.bincount(labels, minlength=a.themes)
    order = np.argsort(-counts)
    out_themes = [dict(id=int(i), size=int(counts[i]), share=float(counts[i] / len(labels)),
                       titles=titles_by_theme[i]) for i in order]

    res = dict(rows=int(len(X)), dims=int(X.shape[1]), read_s=read_s, themes_k=a.themes, themes_s=theme_s,
               themes_iters=int(themes.n_iter_), index_k=a.index_k, index_s=index_s, index_iters=int(index.n_iter_),
               themes=out_themes,
               machine=subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip())
    if a.compare:
        from sklearn.cluster import KMeans as SK
        sub = X[:200_000]
        t = time.perf_counter()
        SK(n_clusters=a.themes, n_init=1, max_iter=themes.n_iter_, random_state=0).fit(sub)
        res["sklearn_rows"], res["sklearn_s"] = len(sub), time.perf_counter() - t
        t = time.perf_counter()
        KMeans(n_clusters=a.themes, max_iter=themes.n_iter_, tol=0.0, random_state=0).fit(sub)
        res["ours_same_rows_s"] = time.perf_counter() - t
        print(f"scikit-learn on {len(sub):,} embeddings: {res['sklearn_s']:.1f}s; ours: {res['ours_same_rows_s']:.2f}s")

    (ROOT / "demos" / "embeddings_results.json").write_text(json.dumps(res, indent=1))
    print()
    for th in out_themes:
        print(f"  {th['share']*100:5.1f}%  {', '.join(th['titles'][:4])}")


if __name__ == "__main__":
    main()
