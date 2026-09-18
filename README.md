# kmeans

Exact k-means (Lloyd's algorithm) on Apple Silicon GPUs, built on MLX custom Metal kernels.
Target machine: Apple M5 Max (18 CPU / 40 GPU cores, 128 GB unified memory).

## Goal

Outperform the public k-means libraries available on this machine (scikit-learn, FAISS, PyTorch MPS
implementations) on problems where it counts: realistic shapes, real datasets, identical results.

- **Publish** if ours is at least **1.5x faster** than the fastest valid public implementation across meaningfully
  general use cases (the suite in `bench.py`).
- **Otherwise** keep it as an internal library for the cases where it measurably wins.
- **Correctness first, then performance.** Every path must pass `tests/accuracy.py` (every point's label vs float64,
  centers and inertia vs float64 accumulation) before any speed claim. Commit each verified state.

## When to use this

| Rows | Verdict |
|---|---|
| under ~100k | **Use scikit-learn.** A pass costs 1-2 ms and GPU dispatch dominates: measured 0.4-0.5x of scikit-learn/FAISS. |
| 1M - 10M+ | 2-8x faster than the fastest public library that reaches the same quality, at any dims 4-960 and k 8-8192. |
| 100M+ | The gap widens: 100M x 32 k=1024 converged in 12 s here; scikit-learn and FAISS take 4-9 minutes and land 3.4-3.9x worse, and MiniBatchKMeans did not finish in 84 minutes. |

## Other Macs

Every hot kernel is GPU-bound, so performance tracks GPU cores and memory bandwidth. This machine: 40 GPU cores,
~400-500 GB/s. Rough expectations relative to it (measure yours with `quick_bench.py` rather than trusting these):

| Machine | GPU cores | vs M5 Max |
|---|---|---|
| MacBook Air | ~10 | ~3.5-4x slower |
| MacBook Pro (Pro chip) | ~16-20 | ~2-2.5x slower |
| MacBook Pro (Max chip) | 40 | 1x |
| Mac Studio (Ultra chip) | ~80 | ~1.7-2x faster |

Memory is the harder limit on small machines: data needs rows * dims * 4 bytes and must fit the GPU working set
(about 70% of RAM), so a 16 GB Air handles ~300M rows at 8 dims or ~20M at 128 dims. The speedup over scikit-learn
and FAISS should if anything be *larger* on an Air, since those lose CPU cores too while the GPU/CPU ratio holds.
Portability: the kernels use Metal simdgroup matrices (M1 and later), not the M5-only Neural Accelerator path.

## Files

| File | What |
|---|---|
| `kmeans.py` | Main implementation: GPU nearest-center kernels (rows / pairs) and compensated accumulation (blocks / sorted) |
| `kmeans_simple.py` | Short plain-MLX version for learning |
| `kmeans.mm` | C++/Metal port of an earlier version (plus an 18-core CPU mode); not kept in sync |
| `tests/accuracy.py` | Float64 ground-truth suite; run before committing kernel changes |
| `bench.py` | Head-to-head suite vs public libraries; writes `benchmarks/results.jsonl` and `BENCHMARKS.md` |
| `bench_ann.py` | End-to-end: wall clock to a given clustering quality, and IVF recall@10 on SIFT1M/GIST1M |
| `quick_bench.py` | Self-benchmark for any Mac: machine info, memory bandwidth, passes vs scikit-learn |
| `benchmarks/NOTES.md` | Findings, design decisions and incidents, in order |

## Running

```bash
python3 tests/accuracy.py
.venv/bin/python bench.py --suite
```

`.venv` (gitignored) adds faiss-cpu, torch and fast-pytorch-kmeans on top of the system MLX/NumPy/scikit-learn.
`data/` (gitignored) holds SIFT1M and GIST1M from the INRIA TEXMEX corpus.

**Shared machine:** register heavy runs in `MACHINE.md` (coordination hub). GPU memory is wired RAM — `bench.py` caps
PyTorch MPS allocations after a benchmark kernel-panicked the machine on 2026-09-17 (see `benchmarks/NOTES.md`).
