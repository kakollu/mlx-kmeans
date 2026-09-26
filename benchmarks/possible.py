"""What this machine allows for one exact Lloyd iteration, per suite config, from the architecture and measured rates.

The floor column in BENCHMARKS.md is the cost of the simplest exact pass on the resources our kernels reach. This is
the other direction: given what the hardware measurably does - DRAM 490 GB/s, the simdgroup matrix unit 15.7 TFLOP/s,
the scalar FMA peak 13.8 TFLOP/s, what mx.matmul reaches on each config's own GEMM, the launch and round-trip costs -
what is the least one iteration can take, under each set of assumptions, and how far the measured pass is from it.
All rates from benchmarks/MACHINE-PROFILE.md (September 2026). The converged-regime column carries the visited
centres' cache reads (VISITED x n x k x dims x 2 bytes at ONCHIP), which measurement showed to be that regime's
cost, not the bound traffic. Run: .venv/bin/python benchmarks/possible.py
"""
DRAM = 490e9            # B/s, one stream of reads
COMBINED = 488e9        # B/s, reads and writes together (shared bus)
SIMD = 15.7e12          # FLOP/s, simdgroup_multiply_accumulate, any precision (custom kernels)
SCALAR = 13.8e12        # FLOP/s, scalar fma peak
ROUND_TRIP = 0.175e-3   # s, host round trip (eval + result), 150-200 us
LAUNCH = 5e-6           # s, one kernel launch
# mx.matmul on each config's GEMM (rows x dims @ dims x k), measured 2026-09-25: TFLOP/s reached, fp32 and fp16.
MATMUL = {"geo-trips": (1.0, 2.0), "satellite": (2.0, 3.9), "logs": (6.4, 13.7), "single-cell": (6.5, 12.1),
          "sift-k1024": (19.5, 32.3), "gist-k1024": (41.1, 57.7)}
# rows, dims, k, measured ms per pass, measured ms per converged-regime iteration (None where the bounds do not engage)
SUITE = {"geo-trips": (10_000_000, 4, 256, 9.2, None), "satellite": (10_000_000, 12, 32, 3.2, None), "logs": (10_000_000, 32, 256, 27.8, None),
         "single-cell": (2_000_000, 50, 64, 4.4, None), "sift-k1024": (1_000_000, 128, 1024, 22.8, None), "gist-k1024": (1_000_000, 960, 1024, 176.0, 47.5)}
VISITED = 0.03          # fraction of the n x k distances a converged-regime iteration still evaluates with per-centre bounds (measured 0.3-3.6%)
ONCHIP = 1.6e12         # B/s the visit phase reads centre rows from cache at, scattered (measured on GIST 1M: 52 GB in 33 ms)
VISIT_BYTES = 2         # per element of a visited centre: the float16 screen decides 99.99% of visits

print(f"{'config':>12} | {'measured':>8} | {'exact, our kernels':>18} | {'exact via matmul':>16} | {'converged, bounds fp32/fp16':>27} | {'result-only':>11} | {'small-data floor':>16}")
print(f"{'':>12} | {'ms/pass':>8} | {'ms  (x off)':>18} | {'ms  (x off)':>16} | {'ms  (x off)':>27} | {'ms  (x off)':>11} | {'round trip':>16}")
for name, (n, d, k, measured, converged) in SUITE.items():
    read = n * d * 4 / DRAM
    gemm = 2.0 * n * k * d
    exact = max(read, gemm / SIMD)                                          # one read, all distances on the matrix unit
    f32, f16 = MATMUL[name]
    via_mm = max(read, gemm / (f32 * 1e12)) if f32 * 1e12 > SIMD else exact  # verified candidates from the fast matmul
    bt32, bt16 = n * k * 8 / COMBINED, n * k * 4 / COMBINED                  # read + rewrite of the bounds
    own = 2.0 * n * d / SCALAR
    visits = VISITED * n * k * d * VISIT_BYTES / ONCHIP                       # the visited centres, read from cache a row at a time
    conv32 = max(read, bt32, VISITED * gemm / SIMD + own, visits)
    conv16 = max(read, bt16, VISITED * gemm / SIMD + own, visits)
    conv32, conv16 = min(conv32, exact), min(conv16, exact)                 # bounds are only taken where they pay
    result_only = max(read, gemm / (max(f16, f32) * 1e12)) if max(f16, f32) * 1e12 > SIMD else exact
    floor_small = ROUND_TRIP + LAUNCH
    m = measured / 1e3
    mc = (converged if converged is not None else measured) / 1e3    # the converged column against the converged iteration
    print(f"{name:>12} | {measured:8.1f} | {exact*1e3:6.2f}  ({m/exact:4.1f}x) | {via_mm*1e3:6.2f}  ({m/via_mm:4.1f}x) | "
          f"{conv32*1e3:6.2f} / {conv16*1e3:5.2f}  ({mc/conv32:4.1f}x/{mc/conv16:4.1f}x{' vs ' + str(converged) + ' ms' if converged else ''}) | {result_only*1e3:5.2f} ({m/result_only:4.1f}x) | {floor_small*1e3:5.2f} ms")
