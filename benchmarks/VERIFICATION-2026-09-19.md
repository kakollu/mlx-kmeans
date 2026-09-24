# Benchmark verification — September 19, 2026

Completed just after midnight September 20 PDT; directory names identify the September 19 start date.

## Scope and machine

The existing benchmark harness was rerun at source revision `c5aa109dd77882bef4c399a731def31bf96594fa`.
This was not a separate implementation of k-means. The implementation was unchanged;
the accuracy test's slice/chunk configuration was repaired before testing. Results are specific to this machine.

Subsequent changes add file parsing/export and correct final-inertia scoring for restart selection in the public
estimator. Those changes were covered by six file/estimator tests but are not included in the timings below.
The Metal kernels are unchanged. For current file usage, see `docs/USING_YOUR_FILES.md`.

Apple M5 Max, 128 GiB unified memory, 18 CPU cores, macOS 27.0, Python 3.11.9; MLX 0.32.2,
scikit-learn 1.9.0, FAISS 1.15.1, PyTorch 2.14.0, fast-pytorch-kmeans 0.2.2. The completed suite ran on AC power,
normal power mode. The initial partial battery run was stopped after power was connected and is excluded.
No competing training/inference/simulation job was observed. Ordinary macOS/UI background work remained;
sampled other-process CPU reached 248.5% (about 2.5 CPU cores). These are normal-desktop measurements, not an
isolated or thermally controlled laboratory run. GPU contention was not directly instrumented.

## Per-pass results

Same input and starting centers; seed 0; warm-up then three timed runs of five Lloyd iterations. Reported time is
the median per-pass value. Scikit-learn wall time is divided by six to credit its final assignment; this convention
favors scikit-learn but is not a direct measurement of six identical update passes. Initialization and input
preparation are outside the timed loop. Comparators were not thread-count-tuned by this verification.

The fastest successful comparator from this run is shown below. Every successful comparison met the existing
0.01% final-inertia agreement threshold. Similar inertia does not prove identical cluster assignments.
No historical best-time selection was used.

| Workload | Shape | MLX ms/pass | Fastest comparator | Comparator ms/pass | Ratio |
|---|---|---:|---|---:|---:|
| Synthetic GPS shape | 10M × 4, k=256 | 10.678 | FAISS | 71.317 | 6.68× |
| Synthetic satellite shape | 10M × 12, k=32 | 9.584 | sklearn Lloyd | 77.624 | 8.10× |
| Synthetic log shape | 10M × 32, k=256 | 70.888 | sklearn Lloyd | 303.008 | 4.27× |
| Synthetic single-cell shape | 2M × 50, k=64 | 10.646 | sklearn Lloyd | 30.569 | 2.87× |
| Real SIFT1M | 1M × 128, k=1024 | 57.614 | sklearn Lloyd | 193.268 | 3.35× |
| Real GIST1M | 1M × 960, k=1024 | 313.466 | fast-pytorch-kmeans MPS | 697.573 | 2.23× |
| Small synthetic | 100k × 8, k=8 | 0.859 | FAISS | 0.998 | 1.16× |
| Small synthetic | 100k × 32, k=64 | 3.511 | sklearn Lloyd | 1.592 | 0.45× |

The log-shape MPS comparator failed with an allocation error under the harness's safety cap (~32.26 GiB).
It is a failed configuration under that cap, not evidence that MPS cannot run it under every configuration.
Other successful comparators remain in the record. Small-case timings fluctuate considerably; do not infer a
universal row-count crossover from the two cases.

On this M5 Max on AC power, six tested workloads of 1M–10M rows measured 2.2–8.1× faster
normalized passes than the fastest tested successful comparator with matching final inertia.
This confirms the scale of the original performance results. It does not establish performance on all workloads,
all libraries, or other Macs.

## Correctness

The repaired `tests/accuracy.py` passed 16/16 cases. Its multi-slice case now actually uses 60,001-row slices
and a 3,110,800-byte distance budget. Counts matched; no labels exceeded the suite's float32-scale tolerance.
The full log records center and inertia errors and strict float64 label differences. The suite permits center and
inertia errors up to 1e-6 under its stated metrics. This is finite-case validation, not a universal exactness proof.

## Complete-workflow measurements

### Complete training on SIFT1M

Three runs per method, seeds 0, 1, 2; 1M × 128, k=1024. These timings include initialization but exclude loading,
NumPy-to-MLX preparation, final common quality evaluation, index building, and query search. The MLX arm uses the
core training loop, not the newer public API's additional labels pass. Methods use different initialization and
stopping policies: MLX up to 50 iterations at relative-inertia tolerance 1e-4, FAISS and full sklearn 25 iterations,
MiniBatchKMeans the harness's batch size 10,000 and stopping settings. These are configuration comparisons.

| Method | Median training seconds | Observed range | Recall@10 range, nprobe=8 |
|---|---:|---:|---:|
| MLX full-data training | 2.503 | 2.458–2.588 | 84.82–84.89% |
| FAISS sampled | 4.960 | 4.934–5.081 | 84.15–84.22% |
| FAISS full-data | 18.461 | 18.235–18.488 | 84.45–84.67% |
| sklearn full-data | 181.701 | 181.354–187.071 | 84.70–84.96% |
| sklearn MiniBatchKMeans | 12.880 | 10.922–13.297 | 83.51–84.09% |

MLX used 33–34 iterations. Its median inertia was 5.04818e10; full sklearn 5.05126e10 and full FAISS 5.05499e10.
The full-data quality differences are small, and sklearn's best observed recall exceeds MLX's best. These results
do not establish universal quality superiority. Training times support a substantial benefit for these configurations.

### Real transaction workload

One month of NYC yellow taxi trips (January 2015): 12,741,035 source rows; 12,589,541 after the demo's plausibility
filters; six numeric features (distance, duration, fare, tip rate, hour, passengers), k=8, seed 0.
The float32 feature input occupied 302,148,984 bytes. Standardization used float64 means and standard deviations
to avoid reduction error on a large float32 table. No peak-memory measurement was made.

| Stage | Seconds |
|---|---:|
| Read/filter local parquet and construct features | 0.741 |
| Standardize features | 0.538 |
| Public `KMeans.fit`, including labels, 14 iterations | 0.134 |
| Sum of these stages | 1.414 |

This is a single measured run with existing local files; downloads, imports, business interpretation, plots and
reporting are excluded. File-cache state was not controlled. It establishes feasibility on this M5 Max, not an
Air time estimate or a clustering-quality validation. Full counts and timings are in `taxi-business.json`.

### Billion-row regression

The CLI completed 1B × 8 synthetic rows, k=16, with 32 GB of GPU-generated input: 0.7 s generation, 1.7 s training
(three iterations), as rounded by its own log. This CLI does not materialize final labels like the public estimator.
Inertia 8.000078e9 is near the generating distribution's expected noise baseline of 8e9; it does not prove a global
optimum. External process wall time was not measured in this rerun.

### Other machines

Loading, cleaning, feature preparation and export contribute to complete workflow time; a fast clustering pass
alone does not establish that total time.

Only this M5 Max was measured. **MacBook Air estimates are planning estimates, not benchmark results.**
*(Superseded on September 24: six machines including an M5 MacBook Air were measured directly — see
`MACHINES-2026-09-24.md`. The paragraph below stands as the position on the date of this verification.)*
One million rows × 20 float32 features is 80 MB of input; ten million is 800 MB. This arithmetic excludes table
storage, preprocessing copies, labels, and GPU workspace. It is not a claim that a particular Air can fit or finish
a workload within a particular time. No quarter-speed extrapolation or larger CPU-relative advantage is asserted.
An Air trial should record its chip, RAM, power mode, data shape, k, iterations, load/prepare/fit times and memory use.

`air_bench.py` provides a memory-budgeted transaction benchmark with `--plan`, `--cleanup`, and `--cleanup-after`.
It reads one 167.2MiB parquet file in batches, with a default maximum of 5M retained rows. A September 20 M5 Max
battery test completed download, fitting, comparison, report export and cleanup; the record is
`local-20260920-070913-795662.json`. MLX fitting took 0.193s and sklearn 1.315s with different iteration counts.
This validates the script, not Air performance, and does not replace the AC measurements above. Six tests cover
memory planning, bounded reading, download validation and cache cleanup. See the README for usage.

## Evidence and reproduction

- `verification-2026-09-19-ac/environment.json`: machine, revision, Python, AC-power observation.
- `verification-2026-09-19-ac/suite.jsonl` and `suite.log`: new suite measurements and failures.
- `verification-2026-09-19-ac/accuracy.log`: repaired correctness suite.
- `verification-2026-09-19-ac/training.jsonl`: 15 training runs, seeds, times, inertia, recall and search timings.
- `verification-2026-09-19-ac/billion.log`: billion-row CLI regression.
- `verification-2026-09-19-ac/taxi-business.json`: real transaction workflow timings and cluster counts.
- `verification-2026-09-19/`: aborted battery run, excluded from claims.

From the repository with dependencies and datasets installed:

```sh
.venv/bin/python scripts/verify_claims.py
.venv/bin/python -u scripts/verify_followups.py
```

The verification scripts use a fixed dated output directory; archive it before another rerun. They do not change
the historical `results.jsonl`, `ann_results.jsonl`, or generated `BENCHMARKS.md`.

The old 100M × 32 competitor training experiment, full 12-shape sweep, other demo datasets, and optimization
ablation claims were not rerun here. The old 100M experiment explicitly selects random initialization for
scikit-learn, so its quality gap is not an automatic library fallback or a fair general quality ranking.
