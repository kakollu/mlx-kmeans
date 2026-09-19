# mlx-kmeans

**Exact k-means on Apple Silicon GPUs.** Cluster 100M rows in seconds on a MacBook, using every row — no sampling,
no cluster, no cloud. Same answers as scikit-learn, 2–8× faster from 1M rows up.

```bash
pip install git+https://github.com/kakollu/mlx-kmeans.git
```

(Not on PyPI yet — that one line installs it straight from this repository. While the repo is private, use
`git+ssh://git@github.com/kakollu/mlx-kmeans.git` instead, with your GitHub SSH key set up.)

```python
import numpy as np
from mlx_kmeans import KMeans

X = np.random.rand(1_000_000, 32).astype(np.float32)   # your data: (rows, dims) float32
model = KMeans(n_clusters=64).fit(X)

model.cluster_centers_   # (64, 32) - the centers
model.labels_            # (1_000_000,) - which cluster each row belongs to
model.inertia_           # total squared distance - lower is better
```

Requires an Apple Silicon Mac (M1 or later) and Python 3.9+. That's the whole setup.

![Seconds per k-means pass versus the fastest public library](docs/speedup.svg)

---

## For a class or a first run

If you have used `kmeans()` in R or scikit-learn in Python, this is the same algorithm with the same arguments and
the same result — it just runs on the Mac's GPU. Three things worth knowing:

**1. It wants float32.** Call `X.astype(np.float32)` before fitting; float64 arrays are converted anyway, at the
cost of a copy.

**2. Standardise your features first**, exactly as you would anywhere else. k-means measures straight-line distance,
so a column in dollars will drown out a column in miles:

```python
Z = ((X - X.mean(0)) / (X.std(0) + 1e-9)).astype(np.float32)
model = KMeans(n_clusters=8, random_state=0).fit(Z)
centers = model.cluster_centers_ * X.std(0) + X.mean(0)      # back to real units to read them
```

**3. Profile the clusters before believing them** — sizes, the mean of each feature per cluster, and a few rows
nearest each center. In the taxi demo below, two of eight segments turned out to differ by *payment method* rather
than by behaviour: the meter only records a tip when the fare is paid by card.

**Under ~100k rows, use scikit-learn instead.** A pass that small takes 1–2 ms and GPU dispatch dominates; we measure
0.4–0.5× of scikit-learn there. This library is for 1M rows and up.

## What your Mac can handle

Data must fit in memory as `rows × dims × 4` bytes, within roughly 70% of RAM:

| Machine | 8 dims | 32 dims | 128 dims | 1536 dims (embeddings) |
|---|---|---|---|---|
| 16 GB Air | 350M rows | 88M | 22M | 1.8M |
| 36 GB Pro | 790M | 200M | 49M | 4.1M |
| 128 GB Max | 2.8B | 700M | 175M | 14M |

Speed tracks GPU cores: an Air has roughly a quarter of an M5 Max's, so expect roughly a quarter of the speed — and a
*larger* advantage over scikit-learn and FAISS, which lose CPU cores too. `python3 quick_bench.py` prints what your
own machine does.

## Benchmarks

Time for one k-means pass against the fastest public library **whose clustering matches ours** (scikit-learn 1.9,
FAISS 1.15, fast-pytorch-kmeans on Metal), same data and same starting centers, on one M5 Max:

| Problem | Shape | This library | Fastest public | Speedup |
|---|---|---|---|---|
| GPS / trip points | 10M × 4, k=256 | 9.9 ms | FAISS 66.1 ms | **6.7×** |
| Satellite pixels | 10M × 12, k=32 | 9.3 ms | scikit-learn 71.1 ms | **7.7×** |
| Log / feature vectors | 10M × 32, k=256 | 70.0 ms | scikit-learn 290.3 ms | **4.1×** |
| Single-cell (after PCA) | 2M × 50, k=64 | 10.2 ms | scikit-learn 28.9 ms | **2.8×** |
| SIFT1M vectors | 1M × 128, k=1024 | 54.0 ms | fast-pytorch-kmeans 174.6 ms | **3.2×** |
| GIST1M vectors | 1M × 960, k=1024 | 289.9 ms | fast-pytorch-kmeans 608.1 ms | **2.1×** |

A 12-shape sweep (100k–10M rows, 4–960 dims, k 8–8192) gives 2.5–7.8× from 1M rows up, and losses below ~100k.

![Time per pass as the data grows](docs/scaling.svg)

**End to end matters more than per pass.** Training an IVF quantiser on SIFT1M (1M × 128, k=1024) — wall clock to a
finished set of centroids, with recall measured against the ground truth that ships with the dataset:

| Method | Train | Clustering quality | recall@10 |
|---|---|---|---|
| **This library, all rows** | **2.1 s** | **best** | **84.9%** |
| FAISS default (trains on a 262k sample) | 5.0 s | 1.0% worse | 84.2% |
| scikit-learn MiniBatchKMeans | 9.7 s | 1.3% worse | 83.9% |
| FAISS, all rows | 18.1 s | 0.2% worse | 84.5% |
| scikit-learn, all rows | 163.7 s | 0.1% worse | 84.8% |

At 100M × 32, k=1024 the gap widens: **12 s here** against 270 s (scikit-learn) and 532 s (FAISS), and those two land
3.4–3.9× worse because at that size they fall back to random seeding within a fixed iteration budget.
MiniBatchKMeans did not finish in 84 minutes. At 1B × 8: 32 GB of data, 1.6 s of clustering, reaching the theoretical
optimum.

### Check the numbers yourself

```bash
git clone https://github.com/kakollu/mlx-kmeans && cd mlx-kmeans
python3 -m venv --system-site-packages .venv
.venv/bin/pip install . scikit-learn faiss-cpu torch fast-pytorch-kmeans
python3 scripts/get_data.py sift gist      # public benchmark sets, no account needed
.venv/bin/python bench.py --suite          # regenerates BENCHMARKS.md
python3 tests/accuracy.py                  # 16 correctness cases against float64
```

Every table in `BENCHMARKS.md` is generated from `benchmarks/results.jsonl`, which records the machine, the git
commit, CPU load during each timing and every raw measurement. A public library whose final inertia differs from ours
by more than 0.01% is marked *different work* and excluded, rather than quietly counted as a win.

## Where the speed comes from

Not from approximating: every point goes to the center with the smallest float32 distance, verified against a float64
reference. The gains are in how the work is laid out on the GPU.

| Change | Effect |
|---|---|
| **One GPU thread per row** instead of one per 4,096-row block | 9–43× at large k × dims. GPUs run threads in lockstep, and a thread looping over thousands of rows throws that away. |
| **Cache-local cluster totals** (rows sorted by cluster, summed in small blocks) | GIST1M totals: 4.16 s → 0.053 s. That step had been 89% of a pass. |
| **Metal SIMD-group matrix instructions** for distances | 3× at high dims, and IEEE-accurate — unlike the Neural Accelerator matmul behind MLX's `@` and PyTorch MPS, which is exact on integers but drifts ~12,000× float32 epsilon on real data. |
| **Greedy k-means++ seeding on the GPU** | Initialization 18.1 s → 0.2 s at k=1024, and roughly 10× fewer iterations to converge. |
| **Unified memory** | 32 GB of data with no host-to-GPU copy, which is why sampling stops being necessary here. |

`benchmarks/NOTES.md` is the full record, including what *didn't* work: cache tiling (1.4×, so cache misses weren't
the cause), branch-free argmin (no effect), triangle-inequality pruning (78% of centers survive at 960 dims), and a
matmul hybrid abandoned on precision grounds.

## Accuracy

`tests/accuracy.py` checks, for every combination of code paths: each point's label against a float64 reference
(exact, up to float32 resolution), cluster centers against float64 accumulation (~1e-13 relative), and empty-cluster
relocation step by step against scikit-learn's own implementation. Cases include overlapping clusters, duplicate
points, duplicate centers, k larger than the number of true clusters, 960 dimensions, large coordinate offsets, and
data split across several GPU buffers. 16/16 pass.

## Demos on real public data

```bash
python3 scripts/get_data.py taxi && .venv/bin/python demos/taxi_segments.py --k 8 --months 8
```

| Demo | Data | Clustered in |
|---|---|---|
| `demos/taxi_segments.py` | 98.5M NYC taxi trips (TLC, 2015) | 0.67 s |
| `demos/satellite_landcover.py` | 96M Sentinel-2 pixels over New York | 0.70 s |
| `demos/embeddings_themes.py` | 1M OpenAI embeddings (1536 dims) | 2.0 s themes, 12.8 s for a 1024-cluster index |

![Unsupervised land cover from one Sentinel-2 scene](docs/landcover.png)

Land cover from 96M pixels with no labels at all: three vegetation densities, two water classes, built-up and bare
ground, each identifiable from its own vegetation and water index. 0.7 seconds on a laptop.

## API

```python
KMeans(n_clusters=8, max_iter=300, tol=1e-4, random_state=None, verbose=False, accumulate="auto")
  .fit(X) / .predict(X) / .fit_predict(X)
  .cluster_centers_  .labels_  .inertia_  .n_iter_
```

`X` may be a NumPy array, an MLX array, or a list of MLX arrays for data larger than one GPU buffer. Empty clusters
are relocated exactly as scikit-learn does. `accumulate="atomic"` is faster (1.7–4.5× on the accumulation step) but
not bit-reproducible, so it is off by default.

For very large runs, `python3 -m mlx_kmeans --rows 1_000_000_000 --dims 8 --k 16` generates and clusters data without
materialising it in NumPy first.

## License

MIT. Built with [MLX](https://github.com/ml-explore/mlx).
