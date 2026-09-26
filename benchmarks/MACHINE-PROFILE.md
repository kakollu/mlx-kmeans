# Apple M5 Max, measured

Everything here was measured on this machine (M5 Max, 40 GPU cores, 128 GB, macOS 27.0) by the experiments
in `NOTES.md`, not taken from a specification. The point of collecting it in one place is that several
numbers we had been assuming were wrong, and the wrong ones sent work in wrong directions for a day.

## The shape of the machine

`applegpu_g17s`, 40 GPU cores, 128 GB unified memory of which Metal will recommend 112.5 GB as a working
set, 80.6 GB largest single buffer. Three facts about it explain almost every result in `NOTES.md`.

**1. There are two multipliers, and a custom kernel can only reach the slower one.** A loop of
`simdgroup_multiply_accumulate` with no memory traffic at all saturates at **15.7 TFLOP/s**. MLX's `@` on
the same shapes reaches 17-36 TFLOP/s in fp32 and 28-52 in fp16/bf16 - above that ceiling, so it is not
running the same instruction. Nothing written in Metal will match `mx.matmul` on throughput; the gap has to
be closed by doing less work or moving less memory, not by writing a better multiply.

**2. Memory is a cliff, not a slope.** Read bandwidth against working-set size, with the total bytes read
held constant so dispatch cost cancels:

| Working set | 128 KB | 512 KB | 2 MB | 8 MB | 32 MB | 128 MB | 512 MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| GB/s | 1328 | 1954 | **2163** | 1772 | 1192 | 485 | 407 |

On-chip it is **about 2 TB/s**; past roughly 32 MB it falls to **~400-480 GB/s**. That is a 4-5x cliff, and
it is the single most useful number here. It is why never materialising the n x k distance matrix was worth
1.2-1.8x (that matrix is gigabytes, so every byte of it is DRAM traffic), why re-reading the centre table
costs nothing (it is kilobytes and lives above the cliff), and why keeping segment totals in registers
instead of device memory was worth 5.6x.

**3. A round trip to the host costs 150-200 microseconds; a launch costs 4-5.** Those are different
things and were conflated for a day. An `eval` - queue the work, wait for it - is 150-196 us. Queuing a
kernel or an MLX op without waiting is **~4-5 us** (measured over chains of 100-1000 dependent launches,
one eval at the end). So chunking a computation finely is nearly free *as long as nothing waits per chunk*:
the fp16 assignment at 960 dims runs at 19.5 ms whether it is cut into 4096-row or 262144-row chunks, when
each chunk is not evaluated. Cut into 4096-row chunks *with* an eval each, it is 38.5 ms. The earlier
"smaller chunks are monotonically worse" result was measuring the per-chunk eval, not the chunking.

**3a. Every round trip to the host costs 150 microseconds.** An `eval` of a trivial op, a one-float readback
and a one-kernel dispatch all measure 147-161 us. Building an MLX op in Python costs 1 us. So the floor on a
Lloyd pass is set by how many times it synchronises, not by the host language - which is why rewriting in
C++ would buy ~0% (measured 0-1% of a step at scale, and the remaining fixed cost is round trips).

The corollary of 2 and 3 together is worth stating, because it cost a day to learn the hard way: **cache
blocking only pays inside a kernel.** Shrinking a chunked kernel's working set into the 2 TB/s region makes
it slower, not faster (161.9 ms to 361.1 ms as chunks go 512 MB to 8 MB), because each chunk costs another
150 us dispatch. The same blocking done inside one kernel is what the fused paths do, and it wins.

## Arithmetic ceilings

| What | Measured | How |
|---|---:|---|
| `simdgroup_multiply_accumulate`, fp32 | **15.7 TFLOP/s** | loop with no loads, 8+ independent accumulators, swept over simdgroup count and threadgroup size |
| ...with 1 / 2 / 4 accumulators | 3.0 / 7.4 / 9.1 | the unit is pipelined; 8 chains are needed to fill it |
| Scalar fp32, as the distance kernel uses it | 9.0 achieved | direct form, 98.3 GFLOP in 10.9 ms, and shown issue-bound by the MAC-count ablation |
| MLX `@`, fp32, 128 dims | 17.2 | 262144 x 128 @ 128 x 1024 |
| MLX `@`, fp16, 960 dims, **vs rows** | 27.0 / 51.9 / 56.6 | at M = 16384 / 65536 / 262144 - the rate depends on the row count |
| MLX `@`, fp16/bf16, best shape seen | 62.8 / 63.0 | 65536 x 4096 @ 4096 x 4096 |
| MLX `@`, fp32, 960 dims | 36.2 | 65536 x 960 @ 960 x 1024 |
| MLX `@`, fp16 / bf16, 128 dims | 28.2 / 30.0 | |
| MLX `@`, fp16 / bf16, 960 dims | 52.1 / 52.2 | |
| Irregular gather-and-score (per-row candidate lists) | 2.3-2.7 | the reason pruning schemes struggle here |

The raw scalar ALU peak is deliberately absent: every synthetic FMA-chain loop written to measure it
returned 90-190 TFLOP/s, which is some six times any plausible figure for 40 cores, and the marginal rate
disagreed with itself between iteration counts. The compiler collapses those loops. What can be defended is
the achieved 9.0 above and the 15.7 simdgroup ceiling, both of which were confirmed by making the work scale
and watching the time scale with it.

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
| Read bandwidth, float4 loads, **best of N** | 510 GB/s |
| Write bandwidth | 435 GB/s |
| Copy, 1 GB in + 1 GB out | **488 GB/s combined** - read and write share one bus, they do not add |
| MLX elementwise / dtype convert, combined | 508-516 GB/s |

**Medians drift.** The same read probe returned medians of 368-505 GB/s across four trials a few seconds
apart, with worst cases down to 249, while the best of each trial sat at 510-516. Nothing heavy was running;
the GPU is shared with the display, and the desktop app's own rendering was 24-30% of a core. So: **ceilings
are best-of-N; comparisons are interleaved A/B**; a median from one run is not a number. The 917 ms - 1016 ms
- 892 ms sequence on the same gist benchmark in one afternoon is what ignoring this looks like.

**AC against battery** (checked 2026-09-25): the fp16 matmul rate, the exact pass and the approximate pass
were the same to within 2% on both. Bandwidth probes are dominated by the contention above either way.

### Older memory numbers

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

## How to compute a floor on this machine

The recipe that decided when to stop optimising, written so it can be reused:

1. **Split the pass into stages and name the unit each one is bound by.** Multiply (which multiplier?
   custom Metal is capped at 15.7 TFLOP/s, `mx.matmul` at 52-57 fp16 / 38-44 fp32 for our shapes),
   streaming reads (490 GB/s combined), on-chip work (2 TB/s if it fits in ~32 MB), host round trips
   (150-200 us each), pure launches (4-5 us each).
2. **Floor each stage from its unit**, then sum. Bytes that must be written and read again count twice
   against the shared 490 GB/s.
3. **Measure each stage separately with a sync after it**, best-of-N. Compare stage by stage, not total to
   total - the total hides which stage is off.
4. **Name the scope.** A floor is conditional on which units are reachable. Ours: custom kernels cannot
   reach the fast multiplier; `mx.matmul` cannot fuse a reduction. Either changing moves the floor.
5. **Stop at the measurement's resolution, not at zero.** With ~25% median drift under UI load and ~3%
   on the multiply, a 10-15% gap on a 27 ms pass is inside what an A/B can resolve. Pieces below ~1.5% each
   are not findable on this machine without dedicated counters.

Applied at 960 dims (GIST 500k rows, k=1024): see `NOTES.md`, "Optimal at 960 dims".

## Threadgroup memory and occupancy (measured September 25, 2026)

A streaming kernel (each thread sums a strided slice of 256 MB) with a threadgroup buffer of the stated size
declared and touched once. What the buffer costs is occupancy: the per-core pool is small, so every KB a
threadgroup declares keeps other threadgroups off the core.

| threads per threadgroup | 1 KB | 4 KB | 8 KB | 12 KB | 16 KB | 24 KB | 32 KB |
|---|---|---|---|---|---|---|---|
| 32 | 333 | 230 | 119 | 110 | 76 | 69 | 57 |
| 64 | 350 | 343 | 153 | 239 | 194 | 130 | 100 |
| 128 | 371 | 366 | 370 | 258 | 196 | 140 | 116 |
| 256 | 334 | 365 | 314 | 202 | 263 | 142 | 224 |

GB/s streamed. Rule that fits the table: keep threadgroup memory under about 64 bytes per resident thread
(8 KB per 128 threads) and bandwidth holds; at 128 bytes per thread it is half, at 256 a third. A
per-simdgroup accumulator of k x (dims+2) floats - 13 KB at k=64, 50 dims - therefore cannot live in
threadgroup memory without collapsing the kernel around it; this is the measured reason the fused
single-read kernel is limited to small k x dims, and why a one-hot tile accumulation only wins once its
footprint is a few KB (see NOTES, "The fused single-read kernel, second attempt").

## Scalar FMA peak and what the matrix hardware delivers per shape (measured September 25, 2026)

Scalar `fma` peak, eight independent chains per thread, no memory traffic: **13.8 TFLOP/s** - within 12% of the
simdgroup matrix ceiling, so on this GPU a scalar distance loop is not slow because it is scalar; it is slow
when its loop overhead (the compare and select per centre, the loads from cache) crowds out the multiplies.
The rows kernel at 10M x 4, k=256 runs at 18% of it.

`mx.matmul` on each benchmark config's own GEMM (rows x dims @ dims x k), TFLOP/s reached. This is the hardware
our custom kernels cannot reach (38-63 TFLOP/s on large GEMMs) - and it only exists at large k x dims:

| config | GEMM | fp32 | fp16 | simdgroup ceiling |
|---|---|---|---|---|
| geo-trips 10M x 4, k=256 | 20 GFLOP | 1.0 | 2.0 | 15.7 |
| satellite 10M x 12, k=32 | 8 GFLOP | 2.0 | 3.9 | 15.7 |
| logs 10M x 32, k=256 | 164 GFLOP | 6.4 | 13.7 | 15.7 |
| single-cell 2M x 50, k=64 | 13 GFLOP | 6.5 | 12.1 | 15.7 |
| SIFT1M 1M x 128, k=1024 | 262 GFLOP | 19.5 | 32.3 | 15.7 |
| GIST1M 1M x 960, k=1024 | 1966 GFLOP | 41.1 | 57.7 | 15.7 |

Below about 128 dims the matmul path is slower than a simdgroup kernel; above it, 1.2-2.6x faster in float32
and 2-3.7x in float16. `benchmarks/possible.py` turns these into per-config bounds.

## On-chip bandwidth for scattered row reads, by working set (measured September 25, 2026)

The bounds step's visit pattern: one simdgroup reads a random 1.9 KB row (960 float16) lane-parallel and
reduces it, 64 rows per simdgroup, 32k simdgroups, rows drawn from a table of the stated size with a proper
hash. GB/s delivered:

| working set | 0.1 MB | 0.5 | 0.9 | 1.4 | 1.9 | 2.8 | 3.8 | 5.6 | 7.5 | 11 | 15 | 22 | 30 | 60 | 120 | 480 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GB/s | 1200 | 1460 | 1470 | 1980 | 1830 | 1660 | 1970 | 2130 | 2540 | 2240 | 1930 | 1450 | 1300 | 1110 | 850 | 610 |

Run-to-run spread is about 20% (a 1.9 MB table read at 1830 in one run and 2690 in the next). Three things
to take from it: the pattern's ceiling is about 2.5 TB/s, reached with working sets of 4-11 MB; small tables
are slower, not faster, because many simdgroups then hit the same lines at once; and it falls to DRAM-like
rates only past ~100 MB, with 480 MB still at 610 GB/s because the hot rows stay resident. Row stride does
not matter: 1920, 2048, 2176, 2304 and 2560-byte strides at 1024 rows all read at 2.6-2.7 TB/s, so there is
no set-conflict penalty to pad away. The bounds step's visit phase (52 GB at 1.6 TB/s on GIST 1M, k=1024, a
2 MB float16 centre table) is therefore at 65-80% of what its own access pattern can get.

