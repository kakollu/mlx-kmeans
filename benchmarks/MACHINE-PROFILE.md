# Apple M5 Max, measured

Everything here was measured on this machine (M5 Max, 40 GPU cores, 128 GB, macOS 27.0) by the experiments
in `NOTES.md`, not taken from a specification. The point of collecting it in one place is that several
numbers we had been assuming were wrong, and the wrong ones sent work in wrong directions for a day.

## Arithmetic ceilings

| What | Measured | How |
|---|---:|---|
| `simdgroup_multiply_accumulate`, fp32 | **15.7 TFLOP/s** | loop with no loads, 8+ independent accumulators, swept over simdgroup count and threadgroup size |
| ...with 1 / 2 / 4 accumulators | 3.0 / 7.4 / 9.1 | the unit is pipelined; 8 chains are needed to fill it |
| Scalar fp32, as the distance kernel uses it | ~9.0 of ~11 | direct form, 98.3 GFLOP in 10.9 ms |
| MLX `@`, fp32, 128 dims | 17.2 | 262144 x 128 @ 128 x 1024 |
| MLX `@`, fp32, 960 dims | 36.2 | 65536 x 960 @ 960 x 1024 |
| MLX `@`, fp16 / bf16, 128 dims | 28.2 / 30.0 | |
| MLX `@`, fp16 / bf16, 960 dims | 52.1 / 52.2 | |
| Irregular gather-and-score (per-row candidate lists) | 2.3-2.7 | the reason pruning schemes struggle here |

**The single most important line:** MLX's matmul reaches 2-3x what `simdgroup_multiply_accumulate` can do,
so a custom Metal kernel cannot match `mx.matmul` on throughput however well written. Our tile matmul at
14.7 TFLOP/s is at **94% of its instruction's ceiling**, not 37% of the machine's. A note claiming ~40
TFLOP/s peak kept "improve the GEMM" on the todo list for a day; it was never reachable that way.

The corollary is that exactness has a hardware price at high dimensions. MLX's fast path carries ~5900x eps
of error against our kernel's ~9x, far too much for a candidate bound to survive - a window widened by that
much admits every centre.

## Memory

| What | Measured |
|---|---:|
| Read bandwidth, float4 loads | 467 GB/s |
| Read bandwidth, scalar loads | 336 GB/s |
| Threadgroup memory | 32 KB = 8192 float32 slots |
| Best achieved by an accumulation kernel | 1.1x one pass over the data (GIST1M) |
| `mx.argsort` on uint32 | ~0.6 ns/row, slightly superlinear (1.9 ms at 4M, 6.0 at 10M, 12.5 at 20M) |

## Things the compiler and runtime do that are not obvious

- **An array indexed by a runtime value does not stay in registers.** `uint cand[M]; cand[nc] = c; nc++`
  goes to scratch memory: 11.3 ms against 5.7 ms for the same kernel keeping candidates in named registers,
  on a multiply costing 5.3 ms. This cost more than the work it was protecting.
- **Arrays of simdgroup matrices spill when indexed in two dimensions.** `a[RK][DK]` instead of
  `a0[DK], a1[DK]` - the same 16 matrices - turned a 16.0 ms kernel into 65.1 ms.
- **A per-thread `float xv[D]` spills above about 96 floats**, and the loss is abrupt: 1.91x at 96 dims
  becomes 0.31x at 104.
- **Threadgroup `atomic_float` works; `atomic_ulong` aborts the process**, which rules out deterministic
  64-bit integer accumulation on chip.
- **Atomicity itself costs 1.8x** even with no contention, and fixes no summation order, so it cannot back a
  reproducible result.
- **`grid` is total threads, not threadgroups.** Getting this wrong reports ~10x the hardware peak, which is
  how it was caught.
- **Smaller dispatches are not cheaper.** Shrinking a chunked kernel's working set to fit cache is monotonic
  the wrong way: 161.9 ms to 361.1 ms as the chunk goes 512 MB to 8 MB, because dispatch and synchronisation
  cost more than the traffic saved.
- **N private threadgroup tiles cost a core exactly what N threadgroups do**, so packing simdgroups together
  to raise occupancy does nothing unless they share the tile.

## What this implies for algorithms here

- **Fusion beats materialising**, and that is the one lever that repeatedly paid: never writing the n x k
  distance matrix was worth 1.2-1.8x on a fit, and keeping segment totals in registers rather than device
  memory was worth 5.6x on GIST1M. Both are the FlashAttention trade - spend arithmetic, save bandwidth.
- **Lane assignment decides whether you need atomics.** Giving each lane a row makes 32 lanes collide on one
  cluster slot; giving each lane a dimension makes them collide never, at identical arithmetic throughput.
- **Register capacity, not bandwidth, is what caps the fused kernels** - which is why the one-pass assignment
  path stops at 96 dims and the high-dimensional case still materialises.
- **Distance-bound pruning is weak in high dimensions and strong in moderate ones.** Ordering dimensions by
  variance and using a partial sum as a lower bound leaves 4.1% of centres alive at 128 dims but 12.8% at
  960 - and the survivors must then be scored irregularly, at 2.3-2.7 TFLOP/s, which is what kills it.

## Six machines, same code

See `MACHINES-2026-09-24.md`: from 5 GPU cores to 40, the spread on a low-dimensional pass is 7.6x, and on a
small one only 3.1x - low-dimensional k-means is bandwidth-bound, so core count buys less than it looks.
