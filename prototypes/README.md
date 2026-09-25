# Prototypes

Working code that is measured but not integrated. Each file carries its own measurements and the list of
what would be needed to ship it, so it can be picked up without re-deriving anything.

- `cascade_prune.py` — the standalone prototype of the prefix-pruning cascade, kept from before it was
  integrated into `mlx_kmeans/core.py`. Useful as the simplest readable statement of the idea.
- `fused_online_argmin.py` — Codex's recommended high-dimensional kernel (`meta/ai/CODEX.md`), built as
  specified and measured: fused online argmin, all 32 lanes reducing, 16-64 centres per step, fp32 or fp16
  inputs. It runs at the 15.7 TFLOP/s instruction ceiling and loses to `mx.matmul` + argmin at every dimension
  from 256 to 960 (0.45-0.69x). Kept as the reference for why this direction is closed.
