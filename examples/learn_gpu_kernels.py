#!/usr/bin/env python3
"""Four lessons in GPU kernels, for someone who already knows OpenMP and MPI.

    python3 examples/learn_gpu_kernels.py

Every lesson runs on your Mac and prints real numbers. The kernels here are simplified versions of the ones in
mlx_kmeans/core.py, so when a lesson lands you can go read the production kernel and recognise it.

The translation table, up front:

    OpenMP / MPI                        Metal (and CUDA, which uses different words for the same things)
    -------------------------------     ------------------------------------------------------------------
    #pragma omp parallel for            launching a kernel over a "grid" of threads
    the loop body                       the kernel source: what ONE iteration does
    omp_get_thread_num()                thread_position_in_grid
    ~16 threads                         millions of threads; 96M is normal
    threads are independent             threads run in LOCKSTEP groups of 32 (Metal: simdgroup, CUDA: warp)
    #pragma omp critical / atomic       atomic_fetch_add_explicit
    reduction(+:x) private copies       a private slice of an output buffer per thread, then a reduce pass
    MPI rank on a node                  a thread in a threadgroup (CUDA: thread block)
    MPI_Barrier within a communicator   threadgroup_barrier(mem_flags::mem_threadgroup)
    node-local memory vs remote         threadgroup memory (~32 KB, on-chip) vs device memory (GB, off-chip)

The one idea with no OpenMP analogue is lockstep execution, and it drives most of the design. Lesson 2.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlx.core as mx  # noqa: E402


def rule(n, title):
    print(f"\n{'=' * 78}\nLesson {n}: {title}\n{'=' * 78}")


def timed(f, repeat=20):
    f(); mx.synchronize()                      # warm up: first call compiles the kernel
    ts = []
    for _ in range(repeat):
        t = time.perf_counter(); f(); mx.synchronize(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


# ---------------------------------------------------------------------------------------------------------------
def lesson1():
    rule(1, "A kernel is the body of a parallel for loop")
    print("""
In OpenMP you write the loop and mark it parallel:

    #pragma omp parallel for
    for (int i = 0; i < n; i++) out[i] = a[i] * 2.0f + 1.0f;

On a GPU you write ONLY the body, as if for a single i, and say how many i there are. There is no loop:""")

    source = """
        uint i = thread_position_in_grid.x;       // your omp_get_thread_num(), but it IS the loop index
        out[i] = a[i] * 2.0f + 1.0f;
    """
    kernel = mx.fast.metal_kernel(name="scale", input_names=["a"], output_names=["out"], source=source)

    a = mx.arange(8, dtype=mx.float32)
    out = kernel(inputs=[a], grid=(8, 1, 1), threadgroup=(8, 1, 1),
                 output_shapes=[(8,)], output_dtypes=[mx.float32])[0]
    print(f"\n    in    {np.array(a)}\n    out   {np.array(out)}")
    print("""
    grid=(8,1,1)         how many threads in total - one per element. This is your loop bound.
    threadgroup=(8,1,1)  how many threads share on-chip memory and can sync with each other (lesson 4).

`grid` is a TOTAL thread count, not a block count. Launching 96 million threads is normal and correct;
the hardware schedules them. Do not try to be clever and give each thread a chunk of work - lesson 3.""")


# ---------------------------------------------------------------------------------------------------------------
def lesson2():
    rule(2, "Threads run in lockstep, so branches cost even when not taken")
    print("""
This is the idea with no OpenMP equivalent, and it shapes everything.

Threads execute in groups of 32 that share ONE instruction pointer (Metal: simdgroup; CUDA: warp). When
threads in a group disagree about a branch, the hardware runs BOTH sides and masks off the threads that
should not have taken each one. A two-way branch inside a hot loop can cost you both sides, every time.

So GPU code prefers arithmetic that is always executed over branches that are sometimes skipped. Here is
argmin written both ways - this is exactly the choice in mlx_kmeans/core.py's _ROWS_SRC:""")

    branchy = """
        uint i = thread_position_in_grid.x;
        float best = INFINITY; uint bl = 0;
        for (uint c = 0; c < K; c++) {
            float v = d[i * K + c];
            if (v < best) { best = v; bl = c; }          // divergent: threads disagree on c
        }
        lab[i] = bl;
    """
    branchless = """
        uint i = thread_position_in_grid.x;
        float best = INFINITY; uint bl = 0;
        for (uint c = 0; c < K; c++) {
            float v = d[i * K + c];
            bool lt = v < best;                          // no branch: both updates always run
            best = select(best, v, lt);
            bl = select(bl, c, lt);
        }
        lab[i] = bl;
    """
    n, k = 4_000_000, 64
    d = mx.array(np.random.default_rng(0).standard_normal((n, k)).astype(np.float32))
    mx.eval(d)
    out = {}
    for tag, name, src in [("branchy", "if / branchy", branchy), ("branchfree", "select / branch-free", branchless)]:
        # The kernel name becomes a C symbol, so it must be a valid identifier - no spaces, no punctuation.
        kern = mx.fast.metal_kernel(name=f"argmin_{tag}", input_names=["d"], output_names=["lab"], source=src)
        run = lambda: mx.eval(kern(inputs=[d], template=[("K", k)], grid=(n, 1, 1), threadgroup=(64, 1, 1),
                                   output_shapes=[(n,)], output_dtypes=[mx.uint32])[0])
        out[name] = (timed(run), np.array(kern(inputs=[d], template=[("K", k)], grid=(n, 1, 1),
                     threadgroup=(64, 1, 1), output_shapes=[(n,)], output_dtypes=[mx.uint32])[0]))
    print(f"\n    {n:,} rows x {k} candidates")
    for name, (t, _) in out.items():
        print(f"    {name:<24} {t*1e3:7.2f} ms")
    same = np.array_equal(out["if / branchy"][1], out["select / branch-free"][1])
    print(f"\n    identical answers: {same}")
    print("""
    The gap is data-dependent: a running minimum stops being updated once it is small, so most threads
    eventually agree and the branch predicts well. The branch-free version is the one that cannot surprise
    you on adversarial data. Where it really pays is any branch where threads genuinely disagree.""")


# ---------------------------------------------------------------------------------------------------------------
def lesson3():
    rule(3, "One thread per row, not one thread per chunk of rows")
    print("""
With 16 OpenMP threads you give each thread a big chunk, because threads are expensive. The GPU instinct is
the opposite: threads are nearly free, and a thread looping over many rows serialises work the machine
wanted to do in parallel. It also wrecks memory access - lockstep threads reading ADJACENT addresses get
one wide coalesced load; threads reading addresses far apart get one transaction each.

Same arithmetic, two decompositions. In mlx_kmeans this choice was worth 9-43x:""")

    per_row = """
        uint i = thread_position_in_grid.x, xi = i * D;
        float s = 0;
        for (uint j = 0; j < D; j++) { float t = X[xi + j] - C[j]; s += t * t; }
        out[i] = s;
    """
    per_chunk = """
        uint b = thread_position_in_grid.x, start = b * CHUNK, end = min(start + CHUNK, uint(N));
        for (uint i = start; i < end; i++) {            // this loop is the mistake
            uint xi = i * D;
            float s = 0;
            for (uint j = 0; j < D; j++) { float t = X[xi + j] - C[j]; s += t * t; }
            out[i] = s;
        }
    """
    n, d, chunk = 4_000_000, 32, 4096
    X = mx.array(np.random.default_rng(0).standard_normal((n, d)).astype(np.float32))
    C = mx.array(np.random.default_rng(1).standard_normal(d).astype(np.float32))
    mx.eval(X, C)

    k1 = mx.fast.metal_kernel(name="dist_row", input_names=["X", "C"], output_names=["out"], source=per_row)
    k2 = mx.fast.metal_kernel(name="dist_chunk", input_names=["X", "C"], output_names=["out"], source=per_chunk)
    r1 = lambda: mx.eval(k1(inputs=[X, C], template=[("D", d)], grid=(n, 1, 1), threadgroup=(64, 1, 1),
                            output_shapes=[(n,)], output_dtypes=[mx.float32])[0])
    nb = -(-n // chunk)
    r2 = lambda: mx.eval(k2(inputs=[X, C], template=[("D", d), ("CHUNK", chunk), ("N", n)], grid=(nb, 1, 1),
                            threadgroup=(64, 1, 1), output_shapes=[(n,)], output_dtypes=[mx.float32])[0])
    t1, t2 = timed(r1), timed(r2)
    a = np.array(k1(inputs=[X, C], template=[("D", d)], grid=(n, 1, 1), threadgroup=(64, 1, 1),
                    output_shapes=[(n,)], output_dtypes=[mx.float32])[0])
    b = np.array(k2(inputs=[X, C], template=[("D", d), ("CHUNK", chunk), ("N", n)], grid=(nb, 1, 1),
                    threadgroup=(64, 1, 1), output_shapes=[(n,)], output_dtypes=[mx.float32])[0])
    print(f"\n    {n:,} rows x {d} dims")
    print(f'    {"one thread per row":<26}{t1*1e3:7.2f} ms   ({n:,} threads)')
    print(f'    {f"one thread per {chunk} rows":<26}{t2*1e3:7.2f} ms   ({nb:,} threads)')
    print(f"    {t2/t1:.1f}x slower for identical arithmetic; same answers: {np.allclose(a, b)}")


# ---------------------------------------------------------------------------------------------------------------
def lesson4():
    rule(4, "Reductions: the race is real, and the best fix is to remove the contention")
    print("""
Now the interesting part, and the one that dominated our k-means. Every thread must add its row into a
shared total. In OpenMP you would write reduction(+:total) and stop thinking. On a GPU you have to pick,
and the three options are the three you already know from OpenMP:

    (a) nothing          -> a data race, and unlike OpenMP you will not be warned
    (b) private copies   -> reduction(+:x): each thread accumulates alone, then a second pass combines
    (c) restructure      -> arrange the data so no two threads ever touch the same output

First, proof that the race is real. 1,000,000 threads each do out[0] += 1.0f:""")

    racy = """
        out[0] += 1.0f;                                  // read, add, write - with a million threads racing
    """
    n = 1_000_000
    kern = mx.fast.metal_kernel(name="racy", input_names=["dummy"], output_names=["out"], source=racy)
    dummy = mx.array([0.0], dtype=mx.float32)
    got = float(np.array(kern(inputs=[dummy], grid=(n, 1, 1), threadgroup=(64, 1, 1),
                              output_shapes=[(1,)], output_dtypes=[mx.float32], init_value=0)[0])[0])
    print(f"\n    expected {n:>12,.0f}\n    got      {got:>12,.0f}   <- lost {(1-got/n)*100:.1f}% of the additions")
    print("""
    Every thread read the same stale value and wrote back over its neighbours. No crash, no warning, just a
    quietly wrong number. This is the single most common GPU bug.

    (b) Private copies, then reduce. Exactly reduction(+:x): give each thread its own accumulator so nobody
    shares, then combine. This is the `blocks` path in mlx_kmeans/core.py:""")

    blocks = """
        uint b = thread_position_in_grid.x, start = b * BS, end = min(start + BS, uint(N));
        float acc = 0;                                   // private to this thread: no sharing, no race
        for (uint i = start; i < end; i++) acc += v[i];
        partial[b] = acc;                                // one slot per thread; combined afterwards
    """
    n, bs = 4_000_000, 4096
    v = mx.array(np.ones(n, dtype=np.float32))
    mx.eval(v)
    nb = -(-n // bs)
    kb = mx.fast.metal_kernel(name="blocks", input_names=["v"], output_names=["partial"], source=blocks)
    part = kb(inputs=[v], template=[("BS", bs), ("N", n)], grid=(nb, 1, 1), threadgroup=(64, 1, 1),
              output_shapes=[(nb,)], output_dtypes=[mx.float32], init_value=0)[0]
    total = float(np.array(part).sum())
    print(f"    {nb:,} private accumulators -> total {total:,.0f}  (expected {n:,})  correct: {total == n}")
    print("""
    (c) Restructure. When each thread must write to one of k outputs, (b) needs a private (k x dims) buffer
    PER THREAD. At k=1024 and 960 dims that is ~985,000 floats per thread, and the scattered writes into
    multi-megabyte buffers swamped everything - 89% of a pass on GIST1M.

    The fix was not a better atomic. It was sorting the rows by cluster first, so each thread owns a
    contiguous run of rows that all belong to ONE cluster and writes a single small contiguous result.
    The contention disappears rather than being managed.

        4.16 s -> 0.053 s, same arithmetic.

    That is the general shape of GPU optimisation: you rarely make the arithmetic faster. You rearrange the
    data so the memory system stops fighting you.""")


def main():
    print(__doc__)
    print(f"Device: {mx.default_device()}")
    for fn in (lesson1, lesson2, lesson3, lesson4):
        fn()
    print("""
================================================================================
Where to go next in this repo
================================================================================
    mlx_kmeans/core.py   _ROWS_SRC        lesson 1 + 2 + 3, the real distance kernel
                         _ACCUMULATE_SRC  lesson 4(b), private buffers
                         _SEGMENT_SRC     lesson 4(c), sorted segments - the one that won
                         _ATOMIC_SRC      threadgroup atomics + a barrier (MPI_Barrier for a node)
                         _DOT_TILES_SRC   simdgroup matrix instructions, 32 threads cooperating on an 8x8 tile

Things that bite people coming from OpenMP:
    - Outputs are NOT zero-initialised unless you pass init_value=0.
    - There is no printf worth using. Debug by writing values into an output buffer and reading them back.
    - A wrong answer is far more likely than a crash. Check against NumPy on small inputs, always.
    - Moving data to and from the GPU costs ~0.14 ms each way. Do not cross that boundary in a tight loop.
""")


if __name__ == "__main__":
    main()
