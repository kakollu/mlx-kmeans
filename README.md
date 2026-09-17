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

## Files

| File | What |
|---|---|
| `kmeans.py` | Main implementation: GPU nearest-center kernels (rows / pairs) and compensated accumulation (blocks / sorted) |
| `kmeans_simple.py` | Short plain-MLX version for learning |
| `kmeans.mm` | C++/Metal port of an earlier version (plus an 18-core CPU mode); not kept in sync |
| `tests/accuracy.py` | Float64 ground-truth suite; run before committing kernel changes |
| `bench.py` | Head-to-head suite vs public libraries; writes `benchmarks/results.jsonl` and `BENCHMARKS.md` |
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
