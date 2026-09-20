# Cluster your own business data

Use this to group customers, transactions, products or other rows with numeric features on an Apple Silicon Mac.
It implements Lloyd's k-means. Everything runs locally. This tool handles the clustering; you decide whether the
chosen features and resulting groups are useful for your question.

For example, annual spend, order count and visits could describe customers. A customer ID identifies a row but
does not describe behavior. Keep IDs, names, dates and category codes out of the selected numeric features.
Categorical variables need a considered encoding before use; treating codes as distances is usually inappropriate.

## Install once

```bash
git clone https://github.com/kakollu/mlx-kmeans.git
cd mlx-kmeans
python3 -m venv .venv
.venv/bin/python -m pip install '.[files]'
```

For each later session, run commands from this repository directory with `.venv/bin/python`.
You do not need to download benchmark datasets to cluster your own file. A small synthetic customer example is
included in `examples/customers.csv`; it is for learning the workflow, not measuring performance.

## Inspect, choose features, run

```bash
.venv/bin/python -m mlx_kmeans examples/customers.csv
.venv/bin/python -m mlx_kmeans examples/customers.csv --columns annual_spend orders visits --k 3 --nstart 10
```

Replace `examples/customers.csv` with your file path, quoting paths or column names containing spaces:

```bash
.venv/bin/python -m mlx_kmeans "my customers.xlsx" --sheet Customers --columns "Annual Spend" Orders Visits --k 4
```

CSV and TSV files must be UTF-8 with unique column names in the first row. XLSX uses its first sheet unless you
specify `--sheet`; it reads saved cell values, so recalculate and save formula-based workbooks in Excel first.
Older `.xls` files should be exported as CSV or XLSX. Inspecting a filename prints a five-row preview and does not fit.

| Parameter | Meaning | Default |
|---|---|---|
| `--k` | Number of groups (R's `centers` when given as a count) | 8 |
| `--nstart` or `--n-init` | Independent starts; keep the lowest final inertia | 10 |
| `--max-iter` or `--iter-max` | Maximum iterations per start | 300 |
| `--seed` | Seed for reproducible starts on the same implementation | 0 |
| `--no-standardize` | Use original numeric scales | Standardize each feature |
| `--missing drop` | Explicitly exclude rows with unusable selected values | Error, asking you to clean or choose |
| `--output results-folder` | Where to save the results | New timestamped folder next to the input |

Standardization keeps a large-valued feature from dominating solely because of its units. This tool uses
population standard deviation (`ddof=0`). Constant features contribute no separation and are reported.
R and this implementation can differ in initialization, algorithm and stopping rules, so a shared seed does not
imply identical groups. The optional `--tol` is relative inertia change, not a center-shift tolerance.

## Read the results

- `labeled.csv`: every input row, in the same order, with an added `cluster` column numbered from 1. If that name
  already exists, the added column gets a `kmeans_` prefix. Rows explicitly excluded by `--missing drop` have blank
  labels; other original columns remain. Input files are never overwritten.
- `cluster_summary.csv`: group sizes, shares among clustered rows, and means of your selected features in their
  original units. Inspect these to describe the groups; group numbers are arbitrary labels, not ranks.
- `run.json`: selected features, standardization values, settings, excluded-row count, timings, iterations and
  final inertia. Inertia is measured in the feature space actually fitted, standardized by default.
- `model.npz`: final centers in fitted feature space, together with feature names and preprocessing values.

Output is CSV/JSON/NPZ, not a copy of workbook formatting or formulas. CSV exports preserve source string values
such as leading-zero IDs; when opening them in Excel or R, import identifier columns as text.
Reusing an existing output folder is rejected. Pick a new folder to compare another `k` or feature selection.

Nothing is silently sampled. File rows are read in batches, but the fitting workflow needs the selected dataset
in memory. Installed RAM sets a conservative budget and current availability checks for a busy Mac. Files beyond
that budget stop with guidance to reduce/export the data. This is not an out-of-core solver or a peak-memory guarantee.

## In a notebook

Open `examples/cluster_your_file.ipynb` with a **Python 3 kernel**, or use:

```python
from mlx_kmeans import cluster_file

result = cluster_file(
    "customers.csv",
    columns=["annual_spend", "orders", "visits"],
    k=3,
    n_init=10,
    seed=0,
)
result["summary"]
```

If your notebook has an **R kernel**, you can keep using it and call the installed command from R. From the
repository directory, for example:

```r
system2(".venv/bin/python", c(
  "-m", "mlx_kmeans", "examples/customers.csv",
  "--columns", "annual_spend", "orders", "visits",
  "--k", "3", "--nstart", "10", "--output", "customer-results"
))
groups <- read.csv("customer-results/cluster_summary.csv")
groups
```

The R snippet invokes the same local Python engine; it does not require changing your analysis notebook to Python.
Use `shQuote()` around file paths or column names containing spaces in `system2()` arguments.

## Using arrays directly

Use `KMeans.fit` with prepared NumPy or MLX arrays to integrate clustering into an analysis pipeline.
Individual Lloyd passes and explicit starting centers are available in `core.py`. The file workflow also performs
parsing, validation and export; its total runtime includes those steps. See the benchmark documentation for
the timing boundaries of the measured performance results.
