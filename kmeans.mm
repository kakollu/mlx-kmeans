// K-means (Lloyd's algorithm) on a randomly generated array, in C++ with Metal. Built for 1B rows.
//
// Data lives in shared-memory Metal buffers (100M rows each), so the GPU and CPU read the same bytes
// with no copies. Data is generated on the GPU. Each assignment pass is either:
//   gpu (default)  one fused Metal kernel: distance + argmin + per-cluster sums in a single read of the data
//   cpu            the same loop in C++ across all CPU cores
//
// Build:  clang++ -std=c++17 -O3 -mcpu=native -ffast-math -fobjc-arc -framework Metal -framework Foundation kmeans.mm -o kmeans
// Run:    ./kmeans --rows 1_000_000_000 --dims 8 --k 16 [--backend gpu|cpu] [--check]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <thread>
#include <vector>

static const uint64_t SLICE_ROWS = 100000000;  // keeps kernel indices inside uint32
static const uint32_t GPU_BLOCK = 4096;         // rows per GPU thread
static const uint64_t CPU_BLOCK = 16384;        // rows per CPU work item

struct Args {
    uint64_t rows = 1000000000;
    int dims = 8, k = 16, max_iter = 50;
    double tol = 1e-5;
    uint64_t seed = 0;
    std::string backend = "gpu";
    bool check = false;
};

struct Pass {  // result of one assignment pass
    std::vector<double> sums;  // k * d
    std::vector<int64_t> counts;
    double inertia = 0;
};

static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static const char *SHADER = R"(
#include <metal_stdlib>
using namespace metal;

static inline ulong mix64(ulong z) {  // splitmix64 finalizer: counter-based RNG, no shared state
    z += 0x9E3779B97F4A7C15ul;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
    return z ^ (z >> 31);
}
static inline float u01(ulong h) { return (float(h >> 40) + 0.5f) / 16777216.0f; }

// Row r = true_center[random label] + standard normal noise (Box-Muller).
kernel void gen(device float *X [[buffer(0)]], constant float *T [[buffer(1)]],
                constant ulong &row0 [[buffer(2)]], constant ulong &seed [[buffer(3)]],
                uint i [[thread_position_in_grid]]) {
    ulong base = mix64(seed ^ mix64(row0 + i));
    uint lab = uint(mix64(base) % K);
    for (uint j = 0; j < D; j++) {
        ulong h = mix64(base + 2 * j + 1);
        float u1 = u01(h), u2 = u01(mix64(h));
        X[i * D + j] = T[lab * D + j] + sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI_F * u2);
    }
}

// Each thread owns one block of BS rows and writes its own (K, D+2) row of [count, inertia, sums...],
// so there are no shared writes; the small per-block results are reduced on the CPU in double.
kernel void assign(device const float *X [[buffer(0)]], constant float *C [[buffer(1)]],
                   device float *out [[buffer(2)]], constant uint &n [[buffer(3)]],
                   uint b [[thread_position_in_grid]]) {
    uint start = b * BS, end = min(start + BS, n), W = D + 2, base = b * K * W;
    for (uint c = 0; c < K * W; c++) out[base + c] = 0;
    for (uint i = start; i < end; i++) {
        float best = 3.4e38f; uint bl = 0;
        for (uint c = 0; c < K; c++) {
            float s = 0;
            for (uint j = 0; j < D; j++) { float t = X[i * D + j] - C[c * D + j]; s += t * t; }
            if (s < best) { best = s; bl = c; }
        }
        uint o = base + bl * W;
        out[o] += 1; out[o + 1] += best;
        for (uint j = 0; j < D; j++) out[o + 2 + j] += X[i * D + j];
    }
}
)";

struct Gpu {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> gen, assign;
    std::vector<id<MTLBuffer>> parts, outs;  // data slices and per-slice block results
    std::vector<uint64_t> rows;              // rows in each slice
};

static id<MTLComputePipelineState> pipeline(Gpu &g, id<MTLLibrary> lib, NSString *name) {
    NSError *err = nil;
    auto pso = [g.dev newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&err];
    if (!pso) { fprintf(stderr, "pipeline %s: %s\n", name.UTF8String, err.localizedDescription.UTF8String); exit(1); }
    return pso;
}

static void setup(Gpu &g, const Args &a) {
    g.dev = MTLCreateSystemDefaultDevice();
    g.queue = [g.dev newCommandQueue];
    NSError *err = nil;
    NSString *src = [NSString stringWithFormat:@"#define D %d\n#define K %d\n#define BS %u\n%s",
                                               a.dims, a.k, GPU_BLOCK, SHADER];
    id<MTLLibrary> lib = [g.dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) { fprintf(stderr, "shader: %s\n", err.localizedDescription.UTF8String); exit(1); }
    g.gen = pipeline(g, lib, @"gen");
    g.assign = pipeline(g, lib, @"assign");
    for (uint64_t s = 0; s < a.rows; s += SLICE_ROWS) {
        uint64_t n = std::min(SLICE_ROWS, a.rows - s);
        uint64_t nb = (n + GPU_BLOCK - 1) / GPU_BLOCK;
        id<MTLBuffer> x = [g.dev newBufferWithLength:n * a.dims * sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> o = [g.dev newBufferWithLength:nb * a.k * (a.dims + 2) * sizeof(float) options:MTLResourceStorageModeShared];
        if (!x || !o) { fprintf(stderr, "could not allocate %.1f GB of GPU memory\n", a.rows * a.dims * 4 / 1e9); exit(1); }
        g.parts.push_back(x); g.outs.push_back(o); g.rows.push_back(n);
    }
}

static const float *row_ptr(const Gpu &g, uint64_t r, int d) {
    return (const float *)g.parts[r / SLICE_ROWS].contents + (r % SLICE_ROWS) * d;
}

static void make_data(Gpu &g, const Args &a) {
    std::mt19937_64 rng(a.seed);
    std::uniform_real_distribution<float> uni(-10, 10);
    std::vector<float> T(a.k * a.dims);
    for (auto &t : T) t = uni(rng);
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        uint64_t row0 = 0;
        for (size_t s = 0; s < g.parts.size(); s++, row0 += SLICE_ROWS) {
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:g.gen];
            [e setBuffer:g.parts[s] offset:0 atIndex:0];
            [e setBytes:T.data() length:T.size() * sizeof(float) atIndex:1];
            [e setBytes:&row0 length:sizeof row0 atIndex:2];
            [e setBytes:&a.seed length:sizeof a.seed atIndex:3];
            [e dispatchThreads:MTLSizeMake(g.rows[s], 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            [e endEncoding];
        }
        [cb commit];
        [cb waitUntilCompleted];
    }
}

// k-means++ seeding on a random subsample (full k-means++ over 1B rows is needlessly slow).
static std::vector<float> kmeans_pp_init(const Gpu &g, const Args &a, std::mt19937_64 &rng) {
    const int d = a.dims;
    size_t m = std::min<uint64_t>(200000, a.rows);
    std::vector<float> S(m * d);
    std::uniform_int_distribution<uint64_t> pick(0, a.rows - 1);
    for (size_t i = 0; i < m; i++) std::copy_n(row_ptr(g, pick(rng), d), d, &S[i * d]);
    auto dist2 = [&](size_t i, const float *c) {
        double s = 0;
        for (int j = 0; j < d; j++) { double t = S[i * d + j] - c[j]; s += t * t; }
        return s;
    };
    std::vector<float> C(a.k * d);
    std::copy_n(&S[std::uniform_int_distribution<size_t>(0, m - 1)(rng) * d], d, &C[0]);
    std::vector<double> d2(m);
    for (size_t i = 0; i < m; i++) d2[i] = dist2(i, &C[0]);
    for (int c = 1; c < a.k; c++) {
        size_t chosen = std::discrete_distribution<size_t>(d2.begin(), d2.end())(rng);
        std::copy_n(&S[chosen * d], d, &C[c * d]);
        for (size_t i = 0; i < m; i++) d2[i] = std::min(d2[i], dist2(i, &C[c * d]));
    }
    return C;
}

static Pass assign_gpu(Gpu &g, const Args &a, const std::vector<float> &C) {
    const int k = a.k, d = a.dims, W = d + 2;
    // One command buffer per slice: packing all 10 slices (32 GB) into one buffer measured ~8% slower at 1B rows.
    for (size_t s = 0; s < g.parts.size(); s++) @autoreleasepool {
        uint32_t n = (uint32_t)g.rows[s];
        id<MTLCommandBuffer> cb = [g.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:g.assign];
        [e setBuffer:g.parts[s] offset:0 atIndex:0];
        [e setBytes:C.data() length:C.size() * sizeof(float) atIndex:1];
        [e setBuffer:g.outs[s] offset:0 atIndex:2];
        [e setBytes:&n length:sizeof n atIndex:3];
        [e dispatchThreads:MTLSizeMake((n + GPU_BLOCK - 1) / GPU_BLOCK, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) { fprintf(stderr, "GPU error: %s\n", cb.error.localizedDescription.UTF8String); exit(1); }
    }
    // reduce the per-block float results in double, one thread per slice
    std::vector<std::vector<double>> tot(g.parts.size(), std::vector<double>(k * W));
    std::vector<std::thread> th;
    for (size_t s = 0; s < g.parts.size(); s++)
        th.emplace_back([&, s] {
            const float *o = (const float *)g.outs[s].contents;
            uint64_t nb = (g.rows[s] + GPU_BLOCK - 1) / GPU_BLOCK;
            for (uint64_t b = 0; b < nb; b++)
                for (int q = 0; q < k * W; q++) tot[s][q] += o[b * k * W + q];
        });
    for (auto &t : th) t.join();
    Pass p{std::vector<double>(k * d), std::vector<int64_t>(k), 0};
    for (auto &t : tot)
        for (int c = 0; c < k; c++) {
            p.counts[c] += (int64_t)t[c * W];
            p.inertia += t[c * W + 1];
            for (int j = 0; j < d; j++) p.sums[c * d + j] += t[c * W + 2 + j];
        }
    return p;
}

// CPU inner loop. DC/KC bake dims and k in at compile time (0 = runtime) so the compiler can vectorize.
// Accumulates into a stack buffer and copies out once: writing straight into the shared output array
// made threads interfere and ran 2x slower.
template <int DC, int KC>
static void cpu_block(const float *__restrict X, uint64_t st, uint64_t en, const float *__restrict C, int k_rt, int d_rt,
                      float *out) {
    const int d = DC ? DC : d_rt, k = KC ? KC : k_rt, W = d + 2;
    float dist[KC ? KC : 1024];
    float acc[(KC ? KC : 1024) * ((DC ? DC : 64) + 2)] = {};
    for (uint64_t i = st; i < en; i++) {
        const float *x = X + i * d;
        for (int c = 0; c < k; c++) {
            float s = 0;
            for (int j = 0; j < d; j++) { float t = x[j] - C[c * d + j]; s += t * t; }
            dist[c] = s;
        }
        int bl = 0; float best = dist[0];  // scalar best, not dist[bl]: 2x faster codegen
        for (int c = 1; c < k; c++) if (dist[c] < best) { best = dist[c]; bl = c; }
        float *o = acc + bl * W;
        o[0] += 1; o[1] += best;
        for (int j = 0; j < d; j++) o[2 + j] += x[j];
    }
    std::copy_n(acc, k * W, out);
}

static void cpu_block_any(const float *X, uint64_t st, uint64_t en, const float *C, int k, int d, float *acc) {
    if (d == 8 && k == 16) return cpu_block<8, 16>(X, st, en, C, k, d, acc);
    if (d == 8) return cpu_block<8, 0>(X, st, en, C, k, d, acc);
    if (d == 16) return cpu_block<16, 0>(X, st, en, C, k, d, acc);
    if (d == 4) return cpu_block<4, 0>(X, st, en, C, k, d, acc);
    if (d == 2) return cpu_block<2, 0>(X, st, en, C, k, d, acc);
    cpu_block<0, 0>(X, st, en, C, k, d, acc);
}

static Pass assign_cpu(Gpu &g, const Args &a, const std::vector<float> &C) {
    const int k = a.k, d = a.dims, W = d + 2;
    struct Job { size_t slice; uint64_t st, en; };
    std::vector<Job> jobs;
    for (size_t s = 0; s < g.parts.size(); s++)
        for (uint64_t st = 0; st < g.rows[s]; st += CPU_BLOCK) jobs.push_back({s, st, std::min(st + CPU_BLOCK, g.rows[s])});
    // like the GPU kernel: each job fills its own float slot (no shared writes), reduced in double afterwards
    std::vector<float> out_buf(jobs.size() * k * W, 0.0f);
    float *out = out_buf.data();
    const Job *job = jobs.data();
    const float *cen = C.data();
    dispatch_apply(jobs.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
        cpu_block_any((const float *)g.parts[job[i].slice].contents, job[i].st, job[i].en, cen, k, d,
                      &out[i * k * W]);
    });
    Pass p{std::vector<double>(k * d), std::vector<int64_t>(k), 0};
    std::vector<double> tot(k * W);
    for (size_t i = 0; i < out_buf.size(); i++) tot[i % (k * W)] += out[i];
    for (int c = 0; c < k; c++) {
        p.counts[c] = (int64_t)tot[c * W];
        p.inertia += tot[c * W + 1];
        for (int j = 0; j < d; j++) p.sums[c * d + j] = tot[c * W + 2 + j];
    }
    return p;
}

// --check: exact double-precision assignment, to validate the float32 kernels.
static std::vector<int64_t> exact_counts(const Gpu &g, const Args &a, const std::vector<float> &C) {
    std::vector<int64_t> counts(a.k);
    const int d = a.dims;
    for (uint64_t r = 0; r < a.rows; r++) {
        const float *x = row_ptr(g, r, d);
        int bl = 0; double best = DBL_MAX;
        for (int c = 0; c < a.k; c++) {
            double s = 0;
            for (int j = 0; j < d; j++) { double t = (double)x[j] - C[c * d + j]; s += t * t; }
            if (s < best) { best = s; bl = c; }
        }
        counts[bl]++;
    }
    return counts;
}

static uint64_t parse_u64(const char *s) {
    std::string t;
    for (; *s; s++) if (*s != '_') t += *s;
    return strtoull(t.c_str(), nullptr, 10);
}

int main(int argc, char **argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string f = argv[i];
        auto val = [&] { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", f.c_str()); exit(1); } return argv[++i]; };
        if (f == "--rows") a.rows = parse_u64(val());
        else if (f == "--dims") a.dims = (int)parse_u64(val());
        else if (f == "--k") a.k = (int)parse_u64(val());
        else if (f == "--max-iter") a.max_iter = (int)parse_u64(val());
        else if (f == "--tol") a.tol = atof(val());
        else if (f == "--seed") a.seed = parse_u64(val());
        else if (f == "--backend") a.backend = val();
        else if (f == "--check") a.check = true;
        else { fprintf(stderr, "usage: kmeans [--rows N] [--dims D] [--k K] [--max-iter N] [--tol T] [--seed S] [--backend gpu|cpu] [--check]\n"); return 1; }
    }
    // the GPU kernels index slices of 100M rows with uint32 and use 4096-row blocks; kmeans.py adapts both
    if (a.k > 1024 || a.dims > 42) { fprintf(stderr, "--k must be <= 1024 and --dims <= 42 (use kmeans.py for larger)\n"); return 1; }
    if (a.backend != "gpu" && a.backend != "cpu") { fprintf(stderr, "--backend must be gpu or cpu\n"); return 1; }

    printf("rows=%llu dims=%d k=%d backend=%s data~%.1f GB\n", a.rows, a.dims, a.k, a.backend.c_str(), a.rows * a.dims * 4 / 1e9);
    Gpu g;
    double t = now();
    setup(g, a);
    make_data(g, a);
    printf("generated data in %.2fs\n", now() - t);

    std::mt19937_64 rng(a.seed);
    auto C = kmeans_pp_init(g, a, rng);
    if (a.check) {
        auto ex = exact_counts(g, a, C);
        Pass pg = assign_gpu(g, a, C), pc = assign_cpu(g, a, C);
        int64_t dg = 0, dc = 0;
        for (int c = 0; c < a.k; c++) { dg += std::llabs(pg.counts[c] - ex[c]); dc += std::llabs(pc.counts[c] - ex[c]); }
        printf("check vs exact double: gpu misassigned %lld, cpu misassigned %lld, inertia gpu/cpu rel diff %.1e\n",
               dg / 2, dc / 2, std::fabs(pg.inertia - pc.inertia) / pc.inertia);
    }

    t = now();
    double prev = -1, inertia = 0;
    for (int it = 1; it <= a.max_iter; it++) {
        double ti = now();
        Pass p = a.backend == "gpu" ? assign_gpu(g, a, C) : assign_cpu(g, a, C);
        inertia = p.inertia;
        std::vector<float> newC = C;
        for (int c = 0; c < a.k; c++) {
            if (p.counts[c] > 0) {
                for (int j = 0; j < a.dims; j++) newC[c * a.dims + j] = (float)(p.sums[c * a.dims + j] / p.counts[c]);
            } else {  // re-seed an empty cluster from a random point
                std::copy_n(row_ptr(g, std::uniform_int_distribution<uint64_t>(0, a.rows - 1)(rng), a.dims), a.dims, &newC[c * a.dims]);
            }
        }
        double shift = 0;
        for (int c = 0; c < a.k; c++) {
            double s = 0;
            for (int j = 0; j < a.dims; j++) { double q = newC[c * a.dims + j] - C[c * a.dims + j]; s += q * q; }
            shift = std::max(shift, std::sqrt(s));
        }
        C.swap(newC);
        printf("iter %3d  inertia %.6e  max_center_shift %.5f  %.3fs\n", it, inertia, shift, now() - ti);
        if (prev >= 0 && std::fabs(prev - inertia) <= a.tol * prev) break;
        prev = inertia;
    }
    printf("done in %.2fs  final inertia %.6e\ncenters (first 5):\n", now() - t, inertia);
    for (int c = 0; c < std::min(5, a.k); c++) {
        for (int j = 0; j < a.dims; j++) printf(" %8.3f", C[c * a.dims + j]);
        printf("\n");
    }
    return 0;
}
