# mlx-kmeans

**K-means clustering on Apple Silicon GPUs.** Cluster local CSV/Excel files or NumPy/MLX arrays, assign rows to
groups, and export labels and cluster summaries. Every Lloyd iteration uses all input rows; initialization uses
a subsample. Data stays on your Mac.

Start with the [file guide](docs/USING_YOUR_FILES.md), [notebook](examples/cluster_your_file.ipynb), or
[Python API](#api). To see it on real data at scale, open
[examples/satellite_landcover.ipynb](examples/satellite_landcover.ipynb) — 96M satellite pixels clustered into
land-cover classes, with saved outputs so it reads without running.

On a 128 GB M5 Max, six measured workloads showed **2.2–8.1× faster passes** than the fastest of the libraries
compared here — scikit-learn, FAISS and fast-pytorch-kmeans — with matching final inertia.
See [benchmark conditions and results](benchmarks/VERIFICATION-2026-09-19.md).

For a local file, after installing `.[files]`:

```bash
python -m mlx_kmeans customers.csv                         # inspect column names
python -m mlx_kmeans customers.csv --columns annual_spend orders visits --k 3 --nstart 10
```

A new results folder contains the original rows with cluster labels, group means in original units, and run
settings. Features are standardized by default; IDs stay out unless explicitly selected. Files are not uploaded.

```bash
pip install git+https://github.com/kakollu/mlx-kmeans.git
```

(Not on PyPI yet — that one line installs it straight from this public repository.)

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

![Seconds per k-means pass versus the fastest of the compared libraries](docs/speedup.svg)

Figure: historical September 18 measurements. The table below contains the September 19 rerun.

---

## Preparing your data

This implements Lloyd's k-means with a Python API resembling scikit-learn's. Initialization, stopping criteria,
and numerical details can produce different clusters. Three things worth knowing:

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

**Small datasets may be faster with scikit-learn.** In the September 19 rerun, 100k rows × 32 features with 64
clusters took 3.51 ms/pass here versus 1.59 ms for scikit-learn. A 100k × 8, k=8 case was roughly tied with FAISS.
The crossover depends on shape, not just row count.

The reason is latency, not arithmetic (measured September 20 on battery power). A GPU round trip costs about
0.14 ms on this machine, and a Lloyd pass here spends roughly 0.9 ms on fixed per-pass cost before it touches
any data — crossing back to the host for the
float64 center update and the convergence check. That floor is flat in row count, so it is invisible at 10M rows
and decisive at 10k. Work only overtakes it above roughly 300k rows at 32 features, after which a pass grows at
about 2 ns per row. Seeding used to add a second fixed cost of one round trip per cluster — the larger of the two
at high k, the smaller at low k, where a few iterations of the pass floor outweigh it — and that one is now
removed (see *Where the speed comes from*). The per-pass floor remains. **If you cluster many small datasets in
a loop, you pay this floor on every iteration — measure before assuming the GPU helps.**

## What your Mac can handle

**Measured machine: 128 GB M5 Max MacBook Pro, on AC power.** The timings below apply to this machine.

**MacBook Air: planning estimates only; not benchmarked here.** A million transactions with 20 numeric features
occupy 80 MB as float32; ten million occupy 800 MB. These are input sizes, not peak memory requirements or tested
Air capacity limits. Loading tables, standardizing features, making copies, storing labels, and GPU workspace all
need additional memory. Start with a representative subset on your Air and measure the complete workflow.

No Air runtime or speedup is established by the M5 Max results. `python3 quick_bench.py` measures your own Mac;
also time `.fit()` on your actual data, since per-pass timings exclude initialization and data preparation.

### Benchmark without administrator access

`no_admin_bench.sh` is for an Apple Silicon Mac where you can use Terminal but do not have Homebrew, Git, or an
administrator password. It uses an available Python 3.9+ when possible. If Python is absent or unusable, it downloads
a private fallback inside the repository. It installs the required packages, runs `quick_bench.py --big`, and saves
a timestamped text report under `benchmarks/`. It does not use `sudo` or change the shell profile. Internet access to
GitHub and Python package downloads is required; the fallback also needs access to Astral. A managed Mac can still
block downloads or execution.

If Git is available:

```bash
git clone https://github.com/kakollu/mlx-kmeans.git
cd mlx-kmeans
./no_admin_bench.sh
```

If Python is available but Git is not, download the public repository as a ZIP using tools included with macOS:

```bash
cd /tmp
curl -L https://github.com/kakollu/mlx-kmeans/archive/refs/heads/main.zip -o mlx-kmeans.zip
ditto -x -k mlx-kmeans.zip .
cd mlx-kmeans-main
./no_admin_bench.sh
```

Copy the printed `benchmarks/no-admin-*.txt` report before leaving a temporary machine. Afterwards, this removes
only the script-owned Python, environment, tools, and download cache; it deliberately preserves the report:

```bash
./no_admin_bench.sh --cleanup
```

This benchmark sizes its largest case from MLX's recommended GPU working set. It measures time per k-means pass,
not complete convergence or maximum safe capacity. To test the existing billion-row regression after setup:

```bash
.no-admin-bench/venv/bin/python -m mlx_kmeans --rows 1000000000 --dims 8 --k 16
```

### Run a simple benchmark on your MacBook Air

```bash
git clone https://github.com/kakollu/mlx-kmeans.git
cd mlx-kmeans
python3 -m venv .venv
.venv/bin/python -m pip install '.[benchmark]'
.venv/bin/python air_bench.py --cleanup-after
```

**The last command runs the benchmark.** It prints both complete clustering times in one table, states which
finished faster, and compares clustering error on the same data. An MLX loss is reported as a loss.
The result is specific to your machine and this workload; an M5 Max advantage does not imply an Air advantage.

Optional: `air_bench.py --plan` only previews the memory budget; it does **not** run anything. Power and memory
details are saved in the report; add `--details` to also print them. The benchmark:

- Chooses up to five million valid taxi transactions using installed RAM, currently available memory, and MLX's
  recommended GPU working set. It budgets at most 25% of RAM, 40% of available memory, or 2 GiB, whichever is lower
  (also limited to 25% of the GPU recommendation). These are conservative working-memory estimates, not guarantees.
- Downloads **one January 2015 taxi file, about 167 MiB**, only after the memory check and a disk-space check.
  It does not download SIFT, GIST, embeddings, or a whole year of taxi records. The complete compressed month is
  stored on disk; only a RAM-sized prefix of valid rows is processed, in batches. This is a benchmark sample,
  not a representative sample for business analysis.
- Measures loading, feature preparation, and complete fitting with labels. Both methods receive the **same rows
  and features**. Their own initialization and stopping rules are included in the complete fitting time, so different
  iteration counts are expected. This answers which finished sooner for this run. Clustering error is calculated
  for both with the same float64 metric outside the timing; lower is better.
- Saves a timestamped JSON report in `benchmarks/local-*.json` with the actual chip, RAM, power settings, row count,
  iterations and timings. This measures the Air itself rather than projecting from the M5 Max.

Plug in power and close heavy apps for comparable results. You can lower the workload further:

```bash
.venv/bin/python air_bench.py --rows 1000000 --memory-mib 512 --cleanup-after
```

For an additional controlled check, run `.venv/bin/python air_bench.py --controlled --cleanup-after`.
This also compares **one Lloyd update plus final labels from identical starting centers**, using the median of three
warmed runs in alternating order. It excludes initialization and GPU input conversion and includes scikit-learn's
internal fit preparation. This diagnostic is separate from the complete training result; it cannot replace it.

Omit `--cleanup-after` to retain the file for repeat runs; the cached file is integrity-checked before reuse.
Clean it up later with `.venv/bin/python air_bench.py --cleanup`. Cleanup removes only this script's owned files
under `data/air-benchmark/`, including incomplete downloads. It preserves reports, other datasets, the repository,
and your virtual environment. `--no-compare` skips scikit-learn. The benchmark does not change your Mac's settings.

## Benchmarks

September 19 AC-power rerun: time per pass against the fastest tested library whose **final inertia matches within
0.01%** (not necessarily identical labels). Same data and starting centers; median of three five-iteration runs.
Scikit-learn time is divided by six to credit its final assignment pass. Initialization is excluded.
The first four workloads are synthetic shapes; SIFT and GIST are real datasets.

| Problem | Shape | This library | Fastest public | Speedup |
|---|---|---|---|---|
| GPS / trip points | 10M × 4, k=256 | 10.68 ms | FAISS 71.32 ms | **6.7×** |
| Satellite pixels | 10M × 12, k=32 | 9.58 ms | scikit-learn 77.62 ms | **8.1×** |
| Log / feature vectors | 10M × 32, k=256 | 70.89 ms | scikit-learn 303.01 ms | **4.3×** |
| Single-cell (after PCA) | 2M × 50, k=64 | 10.65 ms | scikit-learn 30.57 ms | **2.9×** |
| SIFT1M vectors | 1M × 128, k=1024 | 57.61 ms | scikit-learn 193.27 ms | **3.4×** |
| GIST1M vectors | 1M × 960, k=1024 | 313.47 ms | fast-pytorch-kmeans 697.57 ms | **2.2×** |

A historical 12-shape sweep is recorded in `benchmarks/NOTES.md`; it was not repeated in full on September 19.

![Time per pass as the data grows](docs/scaling.svg)

Historical scaling measurements; not rerun on September 19.

**Complete training matters more than per pass.** September 19–20 SIFT1M rerun (1M × 128, k=1024), three seeds per
method. Times include initialization and training, but exclude loading, input conversion, and final label/index
construction. Methods use different initialization and stopping settings; see the verification report.

| Method | Median train time | Observed train range | recall@10 range, nprobe=8 |
|---|---|---|---|
| **This library, all rows (core training)** | **2.50 s** | 2.46–2.59 s | 84.82–84.89% |
| FAISS default (trains on a 262k sample) | 4.96 s | 4.93–5.08 s | 84.15–84.22% |
| scikit-learn MiniBatchKMeans | 12.88 s | 10.92–13.30 s | 83.51–84.09% |
| FAISS, all rows | 18.46 s | 18.23–18.49 s | 84.45–84.67% |
| scikit-learn, all rows | 181.70 s | 181.35–187.07 s | 84.70–84.96% |

Full-data quality is similar; these runs do not establish universal quality superiority. The billion-row CLI
regression also completed: 1B × 8 synthetic rows, k=16, 0.7 s generation and 1.7 s training on this M5 Max.
That CLI generates input directly on the GPU and does not return final labels.

Historical synthetic 100M × 32, k=1024 runs recorded approximately 12 s here, 270 s for scikit-learn, and 532 s
for FAISS. These are different training configurations: the harness explicitly chose random initialization for
scikit-learn above 20M rows and capped competitors at 25 iterations. Their 3.4–3.9× higher inertia does not
establish an inherent library quality disadvantage. These large competitor runs were not repeated on September 19.
The historical 84-minute MiniBatchKMeans run was stopped without finishing.

### Check the numbers yourself

```bash
git clone https://github.com/kakollu/mlx-kmeans && cd mlx-kmeans
python3 -m venv --system-site-packages .venv
.venv/bin/pip install . scikit-learn faiss-cpu torch fast-pytorch-kmeans
python3 scripts/get_data.py sift gist      # public benchmark sets, no account needed
.venv/bin/python bench.py --suite          # regenerates BENCHMARKS.md
.venv/bin/python tests/accuracy.py         # 16 correctness cases against float64
```

Every table in `BENCHMARKS.md` is generated from `benchmarks/results.jsonl`, which records the machine, the git
commit, sampled background CPU load, and median/minimum/maximum timings. A public library whose final inertia differs from ours
by more than 0.01% is marked *different work* and excluded, rather than quietly counted as a win.

## Where the speed comes from

The assignment kernels evaluate nearest centers in float32, with finite-case validation against a float64
reference. The gains are in how the work is laid out on the GPU. The optimization comparisons below are historical
experiments recorded in `benchmarks/NOTES.md`, not fresh ablations in the September 19 verification.

| Change | Effect |
|---|---|
| **One GPU thread per row** instead of one per 4,096-row block | 9–43× at large k × dims. GPUs run threads in lockstep, and a thread looping over thousands of rows throws that away. |
| **Cache-local cluster totals** (rows sorted by cluster, summed in small blocks) | GIST1M totals: 4.16 s → 0.053 s. That step had been 89% of a pass. |
| **Metal SIMD-group matrix instructions** for distances | Historical high-dimensional experiments measured a ~3× gain. The default path refines candidate distances directly; the notes describe precision problems observed with the tested alternative matmul paths. |
| **Greedy k-means++ seeding on the GPU** | Initialization 18.1 s → 0.2 s at k=1024, and roughly 10× fewer iterations to converge. |
| **Seeding without a host sync per round** | Each of the k rounds does little work, so waiting for it made seeding cost k GPU round trips regardless of data size — 16 ms of a 19 ms fit at k=64 on 1k rows. Leaving the rounds unevaluated lets MLX pipeline them: 2–4× faster seeding, and the centers are bit-identical across k=8–1024 because only the timing of evaluation changed. What it saves is a fixed few tens of milliseconds per fit, so complete fits measured 1.1–1.9× faster between 1k and 100k rows and the gain fades to a few percent at millions of rows, where the fit is long enough that seeding was never the problem. |
| **GPU-generated input** | The billion-row CLI creates data on the GPU, avoiding a NumPy input copy. NumPy input through the public API can require additional storage. |

`benchmarks/NOTES.md` is the full record, including what *didn't* work: cache tiling (1.4×, so cache misses weren't
the cause), branch-free argmin (no effect), triangle-inequality pruning (78% of centers survive at 960 dims), and a
matmul hybrid abandoned on precision grounds.

## Accuracy

`tests/accuracy.py` compares assignment paths and deterministic accumulation paths against a float64 reference;
it also tests rows/atomic accumulation where supported. Labels must be optimal within the stated float32-scale
tolerance; center and inertia errors must be within 1e-6 under the suite's metrics. Empty-cluster relocation is
checked against an independent reference, with one case also checked against scikit-learn. The multi-buffer
test's configuration wiring was repaired on September 19. These finite tests are evidence, not a proof for all inputs.
The repaired suite passed **16/16 cases** on the measured M5 Max; the verification directory contains the full log.

## Demos on real public data

**Fresh business-workflow measurement:** January 2015 NYC taxi transactions, 12.59M retained rows × 6 features,
eight clusters: **0.134 s for public `KMeans.fit`, including labels**, and **1.414 s including local file loading,
filtering, feature construction and standardization**. This single M5 Max run excludes downloads, imports and
report generation; file-cache state was not controlled. See the verification report for stage timings. It is not
a MacBook Air projection.

```bash
python3 scripts/get_data.py taxi && .venv/bin/python demos/taxi_segments.py --k 8 --months 8
```

Historical demo measurements below exclude data loading and preparation and predate the API's eager label
calculation. They should not be read as current complete workflow timings.

| Demo | Data | Historical clustering time |
|---|---|---|
| `demos/taxi_segments.py` | 98.5M NYC taxi trips (TLC, 2015) | 0.67 s |
| `demos/satellite_landcover.py` | 96M Sentinel-2 pixels over New York | 0.70 s |
| `demos/embeddings_themes.py` | 1M OpenAI embeddings (1536 dims) | 2.0 s themes, 12.8 s for a 1024-cluster index |

![Unsupervised land cover from one Sentinel-2 scene](docs/landcover.png)

Land cover from 96M pixels with no labels at all: three vegetation densities, two water classes, built-up and bare
ground, interpreted using vegetation and water indices. These are unsupervised interpretations, not validated
land-cover classifications.

Measured on September 20 with the complete current API, labels included: **95,992,216 pixels × 6 features into 8
classes in 0.9 s**, 16 iterations. *These September 20 figures were taken on battery power, unlike the AC-power
suite above; treat the ratios as sound and the absolute times as a floor, since battery runs are not faster.* Reading the four band files took 5.8 s — clustering the whole scene is now six
times quicker than loading it, which is the point at which the algorithm stops being what limits you. On a matched
10M-pixel slice, scikit-learn took 2.5 s against 0.12 s here. Reproduce with:

```bash
.venv/bin/python scripts/get_data.py sentinel
.venv/bin/python demos/satellite_landcover.py --k 8 --compare
```

Nothing in the run knows what a city or a river is. The separation of Manhattan and the outer boroughs from the
Hudson Valley forest is the clustering finding it in six numbers per pixel. The names in the legend are assigned
afterwards from each cluster's own NDVI and NDWI, and the three vegetation classes split by canopy density rather
than by species — k=8 is a choice, and a different k tells a different story, so vary it before drawing conclusions.

## API

The file workflow is a convenience layer. You can continue to control data preparation, seeding, iteration counts,
restarts, and array placement directly through the API; see `mlx_kmeans/core.py` for individual Lloyd passes and
the Metal kernels. Benchmark initialization, full fitting, and individual passes separately when assessing gains.

```python
KMeans(n_clusters=8, n_init=1, max_iter=300, tol=1e-4, random_state=None,
       verbose=False, spherical=False, accumulate="auto")
  .fit(X) / .predict(X) / .fit_predict(X)
  .cluster_centers_  .labels_  .inertia_  .n_iter_
  KMeans.from_centers(centers)        # a fitted model from centers you saved
```

`X` is anything array-like — NumPy array, pandas DataFrame, list of rows — or a list of MLX arrays for data larger
than one GPU buffer. Empty-cluster relocation follows scikit-learn's approach, but pairing order can differ when
several clusters are empty.

**Picking k.** Passes are cheap enough to sweep instead of guess:

```python
for k in range(2, 21):
    print(k, KMeans(n_clusters=k, random_state=0).fit(Z).inertia_)   # elbow curve over 10M rows in seconds
```

**Restarts.** `n_init=10` runs ten starts and keeps the best; the data stays on the GPU between them, so ten starts
cost roughly ten complete training runs, including initialization and each run's iterations. Different starts can
reach different local minima.

**Embeddings.** `spherical=True` re-normalises centers each iteration, which is the right objective for L2-normalised
vectors compared by cosine similarity (what FAISS calls `spherical=True`).

**Saving a model.** The centers are the model: `np.save("centers.npy", model.cluster_centers_)`, later
`KMeans.from_centers(np.load("centers.npy")).predict(new_rows)`.

`accumulate="atomic"` is faster (1.7–4.5× on the accumulation step) but not bit-reproducible, so it is off by
default.

For very large runs, `python3 -m mlx_kmeans --rows 1_000_000_000 --dims 8 --k 16` generates and clusters data without
materialising it in NumPy first.

## License

MIT. Built with [MLX](https://github.com/ml-explore/mlx).
