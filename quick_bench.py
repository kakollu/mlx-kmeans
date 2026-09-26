#!/usr/bin/env python3
"""What does this k-means do on YOUR Mac? Prints a short report you can share.

Sizes itself to the machine (GPU working set), measures memory bandwidth and k-means passes at a few shapes, and
compares against scikit-learn if it is installed. Takes a couple of minutes and needs only mlx + numpy.

  python3 quick_bench.py            # default: a few representative shapes
  python3 quick_bench.py --big      # add a large shape sized to this machine's memory
"""
import argparse
import platform
import subprocess
import time

import numpy as np

import mlx_kmeans as km

SHAPES = [(1_000_000, 8, 16), (1_000_000, 32, 256), (1_000_000, 128, 1024), (10_000_000, 12, 32)]


def machine():
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    cores = subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True).stdout.strip()
    mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30
    gpu = "?"
    out = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "Total Number of Cores" in line:
            gpu = line.split(":")[-1].strip()
    return chip, int(cores), mem, gpu


def timed(fn, reps=5, warm_s=0.3):
    """Median of reps after warming for at least warm_s: millisecond kernels measured after one call read up to
    2x slow while the GPU clock is still ramping (1.1-2.0 ms for the same 1M x 8 pass, measured)."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < warm_s:
        fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


_STREAM_SRC = """
    uint i = thread_position_in_grid.x, T = threads_per_grid.x;
    float s = 0.0f;
    for (uint j = i; j < N; j += T) s += X[j];
    out[i] = s;
"""


def stream_rate(mx, x):
    """Bytes per second of a plain streaming read of x by a custom kernel - the rate a pass's floor is built on.
    mx.sum reports a reduction rate instead, 265 against 490 GB/s on an M5 Max."""
    kern = mx.fast.metal_kernel(name="stream_read", input_names=["X"], output_names=["out"], source=_STREAM_SRC)
    threads = 1 << 18
    run = lambda: mx.eval(kern(inputs=[x], template=[("N", x.size)], grid=(threads, 1, 1), threadgroup=(256, 1, 1),
                                output_shapes=[(threads,)], output_dtypes=[mx.float32])[0])
    return x.size * 4 / timed(run)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--big", action="store_true", help="add a large shape sized to this machine's memory")
    a = p.parse_args()
    mx = km._mx()
    chip, cores, mem_gb, gpu_cores = machine()
    budget_gb = mx.device_info()["max_recommended_working_set_size"] / 2**30
    print(f"{chip}, {cores} CPU cores, {gpu_cores} GPU cores, {mem_gb:.0f} GB memory "
          f"({budget_gb:.0f} GB usable by the GPU), macOS {platform.mac_ver()[0]}, mlx {mx.__version__}")

    x = mx.zeros((250_000_000 if budget_gb > 16 else 50_000_000,), dtype=mx.float32)
    mx.eval(x)
    gb = x.size * 4 / 1e9
    bw = stream_rate(mx, x) / 1e9
    print(f"memory read bandwidth: {bw:.0f} GB/s streaming ({gb:.1f} GB in {gb/bw*1000:.1f} ms)\n")
    del x
    mx.clear_cache()

    shapes = list(SHAPES)
    if a.big:
        rows = int(min(0.4 * budget_gb * 1e9 / (32 * 4), 200_000_000))
        shapes.append((rows, 32, 1024))
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        KMeans = None
    print(f"{'rows':>12} {'dims':>5} {'k':>6} {'ours s/pass':>12} {'rows/s':>14}" + (f" {'scikit-learn':>13} {'speedup':>8}" if KMeans else ""))
    for rows, dims, k in shapes:
        need = rows * dims * 4 / 1e9
        if need > 0.6 * budget_gb:
            print(f"{rows:>12,} {dims:>5} {k:>6}  skipped: needs {need:.1f} GB")
            continue
        parts = km.make_data_mlx(rows, dims, k, 0)
        C = km.kmeans_pp_init(parts, rows, k, np.random.default_rng(0))
        ours = timed(lambda: km.assign_mlx(parts, C))
        line = f"{rows:>12,} {dims:>5} {k:>6} {ours:>12.4f} {rows/ours:>14,.0f}"
        if KMeans and rows * dims * k <= 4e12:
            X = np.concatenate([np.array(p) for p in parts])
            sk = timed(lambda: KMeans(n_clusters=k, init=C, n_init=1, max_iter=5, tol=0.0, algorithm="lloyd").fit(X), reps=1) / 6
            line += f" {sk:>13.4f} {sk/ours:>7.1f}x"
            del X
        print(line, flush=True)
        del parts
        mx.clear_cache()
    print("\nShare this output - it says what your Mac does per k-means iteration.")


if __name__ == "__main__":
    main()
