#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include "timing.h"
#include "attention_ref.h"

static constexpr int   D          = 64;
static constexpr float INV_SQRT_D = 0.125f;

#define FULL_MASK 0xffffffffu

// ============================================================
//  PREFILL KERNELS
// ============================================================

static constexpr int PNBLOCK = 256;

__global__ void prefill_naive_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    float* __restrict__ score_buf,
    int S)
{
    const int i = blockIdx.x;
    float* sc = score_buf + (long long)i * S;
    extern __shared__ float sred[];

    float my_max = -FLT_MAX;
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        float s = 0.f;
        const float* qi = Q + (long long)i * D;
        const float* kj = K + (long long)j * D;
        #pragma unroll 8
        for (int d = 0; d < D; d++) s += qi[d] * kj[d];
        sc[j] = s * INV_SQRT_D;
        my_max = fmaxf(my_max, sc[j]);
    }
    sred[threadIdx.x] = my_max;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sred[threadIdx.x] = fmaxf(sred[threadIdx.x], sred[threadIdx.x + stride]);
        __syncthreads();
    }
    float gmax = sred[0];

    float my_sum = 0.f;
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        sc[j] = expf(sc[j] - gmax);
        my_sum += sc[j];
    }
    sred[threadIdx.x] = my_sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sred[threadIdx.x] += sred[threadIdx.x + stride];
        __syncthreads();
    }
    float gsum = sred[0];

    for (int j = threadIdx.x; j <= i; j += blockDim.x) sc[j] /= gsum;
    __syncthreads();

    for (int dd = threadIdx.x; dd < D; dd += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j <= i; j++)
            acc += sc[j] * V[(long long)j * D + dd];
        O[(long long)i * D + dd] = acc;
    }
}

// prefill_flash: tile=16, __launch_bounds__(D,4)
static constexpr int PF_TILE = 16;

__global__ void __launch_bounds__(D, 4)
prefill_flash_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int S)
{
    const int i  = blockIdx.x;
    const int dd = threadIdx.x;

    __shared__ float sK[PF_TILE][D];
    __shared__ float sV[PF_TILE][D];
    __shared__ float swarp[2];

    float q_d = Q[(long long)i * D + dd];
    float m = -FLT_MAX, s = 0.f, o = 0.f;

    for (int ts = 0; ts <= i; ts += PF_TILE) {
        int te  = min(ts + PF_TILE, i + 1);
        int tsz = te - ts;

        for (int t = dd; t < tsz * D; t += D) {
            int tk = t / D, td = t % D;
            sK[tk][td] = K[(long long)(ts + tk) * D + td];
            sV[tk][td] = V[(long long)(ts + tk) * D + td];
        }
        __syncthreads();

        for (int tk = 0; tk < tsz; tk++) {
            float p = q_d * sK[tk][dd];
            for (int off = 16; off >= 1; off >>= 1)
                p += __shfl_down_sync(FULL_MASK, p, off);
            if ((dd & 31) == 0) swarp[dd >> 5] = p;
            __syncthreads();
            float score = (swarp[0] + swarp[1]) * INV_SQRT_D;
            float m_new = fmaxf(m, score);
            float corr  = expf(m - m_new);
            float es    = expf(score - m_new);
            s = s * corr + es;
            o = o * corr + es * sV[tk][dd];
            m = m_new;
            __syncthreads();
        }
        __syncthreads();
    }
    O[(long long)i * D + dd] = o / s;
}

// prefill_flash_v2: tile=32 (doubled from flash's 16), scalar loads.
// Optimization: 2x larger tile => 2x fewer outer iterations =>
// 2x fewer tile-load __syncthreads() barriers and loop overhead.
// Deliberately avoids float4 because reinterpret_cast with dynamic
// tile sizes causes the compiler to allocate ~150 registers (vs 56).
// Same load pattern as prefill_flash ensures register count stays low.
static constexpr int PF_TILE2 = 32;

__global__ void __launch_bounds__(D, 4)
prefill_flash_v2_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int S)
{
    const int i  = blockIdx.x;
    const int dd = threadIdx.x;

    __shared__ float sK[PF_TILE2][D];
    __shared__ float sV[PF_TILE2][D];
    __shared__ float swarp[2];

    float q_d = Q[(long long)i * D + dd];
    float m = -FLT_MAX, s = 0.f, o = 0.f;

    for (int ts = 0; ts <= i; ts += PF_TILE2) {
        int te  = min(ts + PF_TILE2, i + 1);
        int tsz = te - ts;

        for (int t = dd; t < tsz * D; t += D) {
            int tk = t / D, td = t % D;
            sK[tk][td] = K[(long long)(ts + tk) * D + td];
            sV[tk][td] = V[(long long)(ts + tk) * D + td];
        }
        __syncthreads();

        for (int tk = 0; tk < tsz; tk++) {
            float p = q_d * sK[tk][dd];
            for (int off = 16; off >= 1; off >>= 1)
                p += __shfl_down_sync(FULL_MASK, p, off);
            if ((dd & 31) == 0) swarp[dd >> 5] = p;
            __syncthreads();
            float score = (swarp[0] + swarp[1]) * INV_SQRT_D;
            float m_new = fmaxf(m, score);
            float corr  = expf(m - m_new);
            float es    = expf(score - m_new);
            s = s * corr + es;
            o = o * corr + es * sV[tk][dd];
            m = m_new;
            __syncthreads();
        }
        __syncthreads();
    }
    O[(long long)i * D + dd] = o / s;
}

// ============================================================
//  DECODE KERNELS
// ============================================================

static constexpr int DNBLOCK = 256;

__global__ void decode_naive_kernel(
    const float* __restrict__ q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ o,
    float* __restrict__ score_buf,
    int C)
{
    extern __shared__ float sred[];
    float my_max = -FLT_MAX;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        float s = 0.f;
        #pragma unroll 8
        for (int d = 0; d < D; d++) s += q[d] * K[(long long)j * D + d];
        score_buf[j] = s * INV_SQRT_D;
        my_max = fmaxf(my_max, score_buf[j]);
    }
    sred[threadIdx.x] = my_max;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sred[threadIdx.x] = fmaxf(sred[threadIdx.x], sred[threadIdx.x + stride]);
        __syncthreads();
    }
    float gmax = sred[0];

    float my_sum = 0.f;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        score_buf[j] = expf(score_buf[j] - gmax);
        my_sum += score_buf[j];
    }
    sred[threadIdx.x] = my_sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sred[threadIdx.x] += sred[threadIdx.x + stride];
        __syncthreads();
    }
    float gsum = sred[0];

    for (int j = threadIdx.x; j < C; j += blockDim.x) score_buf[j] /= gsum;
    __syncthreads();

    for (int dd = threadIdx.x; dd < D; dd += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < C; j++) acc += score_buf[j] * V[(long long)j * D + dd];
        o[dd] = acc;
    }
}

// decode_flash: slice=256, subtile=32
static constexpr int DF_BLOCK   = D;
static constexpr int DF_SLICE   = 256;
static constexpr int DF_SUBTILE = 32;

__global__ void decode_flash_p1_kernel(
    const float* __restrict__ q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ partial_m,
    float* __restrict__ partial_d,
    float* __restrict__ partial_O,
    int C)
{
    const int b  = blockIdx.x;
    const int dd = threadIdx.x;
    int cstart = b * DF_SLICE;
    int cend   = min(cstart + DF_SLICE, C);

    __shared__ float sK[DF_SUBTILE][D];
    __shared__ float sV[DF_SUBTILE][D];
    __shared__ float swarp[2];

    float q_d = q[dd];
    float m = -FLT_MAX, s = 0.f, o = 0.f;

    for (int st = cstart; st < cend; st += DF_SUBTILE) {
        int se  = min(st + DF_SUBTILE, cend);
        int ssz = se - st;
        for (int t = dd; t < ssz * D; t += D) {
            int tk = t / D, td = t % D;
            sK[tk][td] = K[(long long)(st + tk) * D + td];
            sV[tk][td] = V[(long long)(st + tk) * D + td];
        }
        __syncthreads();
        for (int tk = 0; tk < ssz; tk++) {
            float p = q_d * sK[tk][dd];
            for (int off = 16; off >= 1; off >>= 1)
                p += __shfl_down_sync(FULL_MASK, p, off);
            if ((dd & 31) == 0) swarp[dd >> 5] = p;
            __syncthreads();
            float score = (swarp[0] + swarp[1]) * INV_SQRT_D;
            float m_new = fmaxf(m, score);
            float corr  = expf(m - m_new);
            float es    = expf(score - m_new);
            s = s * corr + es;
            o = o * corr + es * sV[tk][dd];
            m = m_new;
            __syncthreads();
        }
        __syncthreads();
    }
    if (dd == 0) { partial_m[b] = m; partial_d[b] = s; }
    partial_O[(long long)b * D + dd] = o;
}

__global__ void decode_flash_p2_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_d,
    const float* __restrict__ partial_O,
    float* __restrict__ o,
    int num_blocks)
{
    const int dd = threadIdx.x;
    float m = -FLT_MAX, s = 0.f, out = 0.f;
    for (int b = 0; b < num_blocks; b++) {
        float m_b = partial_m[b];
        float d_b = partial_d[b];
        float o_b = partial_O[(long long)b * D + dd];
        float m_new = fmaxf(m, m_b);
        float c_old = (m == -FLT_MAX) ? 0.f : expf(m - m_new);
        float c_new = expf(m_b - m_new);
        s   = s   * c_old + d_b * c_new;
        out = out * c_old + o_b * c_new;
        m   = m_new;
    }
    o[dd] = out / s;
}

// decode_flash_v2: same slice=256 as flash (same #blocks => same SM utilization),
// but float4 vectorized K/V loads to reduce load instruction count.
// For memory-bound decode, keeping SM utilization identical while using
// wider loads maximizes HBM bandwidth efficiency.
static constexpr int DF2_SLICE   = 256;
static constexpr int DF2_SUBTILE = 32;

__global__ void
decode_flash_v2_p1_kernel(
    const float* __restrict__ q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ partial_m,
    float* __restrict__ partial_d,
    float* __restrict__ partial_O,
    int C)
{
    const int b  = blockIdx.x;
    const int dd = threadIdx.x;
    int cstart = b * DF2_SLICE;
    int cend   = min(cstart + DF2_SLICE, C);

    __shared__ float sK[DF2_SUBTILE][D];
    __shared__ float sV[DF2_SUBTILE][D];
    __shared__ float swarp[2];

    float q_d = q[dd];
    float m = -FLT_MAX, s = 0.f, o = 0.f;

    for (int st = cstart; st < cend; st += DF2_SUBTILE) {
        int se  = min(st + DF2_SUBTILE, cend);
        int ssz = se - st;

        const float4* Kbase = reinterpret_cast<const float4*>(K + (long long)st * D);
        const float4* Vbase = reinterpret_cast<const float4*>(V + (long long)st * D);
        float4* sK4 = reinterpret_cast<float4*>(&sK[0][0]);
        float4* sV4 = reinterpret_cast<float4*>(&sV[0][0]);
        for (int t = dd; t < ssz * (D / 4); t += (D / 4)) {
            sK4[t] = Kbase[t];
            sV4[t] = Vbase[t];
        }
        __syncthreads();

        for (int tk = 0; tk < ssz; tk++) {
            float p = q_d * sK[tk][dd];
            for (int off = 16; off >= 1; off >>= 1)
                p += __shfl_down_sync(FULL_MASK, p, off);
            if ((dd & 31) == 0) swarp[dd >> 5] = p;
            __syncthreads();
            float score = (swarp[0] + swarp[1]) * INV_SQRT_D;
            float m_new = fmaxf(m, score);
            float corr  = expf(m - m_new);
            float es    = expf(score - m_new);
            s = s * corr + es;
            o = o * corr + es * sV[tk][dd];
            m = m_new;
            __syncthreads();
        }
        __syncthreads();
    }
    if (dd == 0) { partial_m[b] = m; partial_d[b] = s; }
    partial_O[(long long)b * D + dd] = o;
}

// ============================================================
//  VERIFICATION
// ============================================================

static bool verify_prefill(float* d_O, float* h_Q, float* h_K, float* h_V,
                            int S, int check_rows)
{
    int vS = min(S, check_rows);
    float* h_O_ref = (float*)malloc((long long)vS * D * sizeof(float));
    flash_prefill_ref((const float(*)[ATTN_D])h_Q,
                      (const float(*)[ATTN_D])h_K,
                      (const float(*)[ATTN_D])h_V,
                      (float(*)[ATTN_D])h_O_ref, vS);
    float* h_O_gpu = (float*)malloc((long long)S * D * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_O_gpu, d_O, (long long)S * D * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float md = attn_max_diff(h_O_gpu, h_O_ref, (long long)vS * D);
    free(h_O_ref); free(h_O_gpu);
    return md < 1e-3f;
}

static bool verify_decode(float* d_o, float* h_q, float* h_K, float* h_V, int C)
{
    float h_o_ref[D];
    standard_decode_ref(h_q, (const float(*)[ATTN_D])h_K,
                        (const float(*)[ATTN_D])h_V, h_o_ref, C);
    float h_o_gpu[D];
    CUDA_CHECK(cudaMemcpy(h_o_gpu, d_o, D * sizeof(float), cudaMemcpyDeviceToHost));
    return attn_max_diff(h_o_gpu, h_o_ref, D) < 1e-3f;
}

// ============================================================
//  BENCHMARK RUNNERS
// ============================================================

static void bench_prefill(int S)
{
    printf("\n=== PREFILL  S=%d ===\n", S);
    long long elem = (long long)S * D;
    float* h_Q = (float*)malloc(elem * sizeof(float));
    float* h_K = (float*)malloc(elem * sizeof(float));
    float* h_V = (float*)malloc(elem * sizeof(float));
    attn_fill(h_Q, elem, 1);
    attn_fill(h_K, elem, 2);
    attn_fill(h_V, elem, 3);
    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, elem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, elem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, elem * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, elem * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, elem * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, elem * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, elem * sizeof(float), cudaMemcpyHostToDevice));
    double flops = 2.0 * D * (double)S * (S + 1);

    if (S <= 4096) {
        float* d_score_buf;
        CUDA_CHECK(cudaMalloc(&d_score_buf, (long long)S * S * sizeof(float)));
        int smem = PNBLOCK * sizeof(float);
        auto fn = [&]() {
            prefill_naive_kernel<<<S, PNBLOCK, smem>>>(d_Q, d_K, d_V, d_O, d_score_buf, S);
        };
        fn(); CUDA_CHECK(cudaDeviceSynchronize());
        printf("  [verify naive:    %s]\n", verify_prefill(d_O, h_Q, h_K, h_V, S, 64) ? "PASS" : "FAIL");
        bench_gpu("prefill_naive", fn, flops);
        CUDA_CHECK(cudaFree(d_score_buf));
    } else {
        printf("  prefill_naive: skipped (score_buf %.1f GB)\n", (double)S * S * 4 / 1e9);
    }

    {
        auto fn = [&]() { prefill_flash_kernel<<<S, D>>>(d_Q, d_K, d_V, d_O, S); };
        fn(); CUDA_CHECK(cudaDeviceSynchronize());
        printf("  [verify flash:    %s]\n",
               verify_prefill(d_O, h_Q, h_K, h_V, S, (S <= 4096) ? 64 : 8) ? "PASS" : "FAIL");
        bench_gpu("prefill_flash (tile=16)", fn, flops);
    }

    {
        auto fn = [&]() { prefill_flash_v2_kernel<<<S, D>>>(d_Q, d_K, d_V, d_O, S); };
        fn(); CUDA_CHECK(cudaDeviceSynchronize());
        printf("  [verify flash_v2: %s]\n",
               verify_prefill(d_O, h_Q, h_K, h_V, S, (S <= 4096) ? 64 : 8) ? "PASS" : "FAIL");
        bench_gpu("prefill_flash_v2 (tile=32)", fn, flops);
    }

    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
    free(h_Q); free(h_K); free(h_V);
}

static void bench_decode(int C)
{
    printf("\n=== DECODE   C=%d ===\n", C);
    float* h_q = (float*)malloc(D * sizeof(float));
    float* h_K = (float*)malloc((long long)C * D * sizeof(float));
    float* h_V = (float*)malloc((long long)C * D * sizeof(float));
    attn_fill(h_q, D, 1);
    attn_fill(h_K, (long long)C * D, 2);
    attn_fill(h_V, (long long)C * D, 3);
    float *d_q, *d_K, *d_V, *d_o;
    CUDA_CHECK(cudaMalloc(&d_q, D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, (long long)C * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, (long long)C * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_o, D * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_q, h_q, D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, (long long)C * D * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, (long long)C * D * sizeof(float), cudaMemcpyHostToDevice));
    double flops = 4.0 * C * D;

    {
        float* d_score_buf;
        CUDA_CHECK(cudaMalloc(&d_score_buf, C * sizeof(float)));
        int smem = DNBLOCK * sizeof(float);
        auto fn = [&]() {
            decode_naive_kernel<<<1, DNBLOCK, smem>>>(d_q, d_K, d_V, d_o, d_score_buf, C);
        };
        fn(); CUDA_CHECK(cudaDeviceSynchronize());
        printf("  [verify naive:    %s]\n", verify_decode(d_o, h_q, h_K, h_V, C) ? "PASS" : "FAIL");
        bench_gpu("decode_naive", fn, flops);
        CUDA_CHECK(cudaFree(d_score_buf));
    }

    {
        int nb = (C + DF_SLICE - 1) / DF_SLICE;
        float *d_pm, *d_pd, *d_pO;
        CUDA_CHECK(cudaMalloc(&d_pm, nb * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pd, nb * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pO, (long long)nb * D * sizeof(float)));
        auto fn = [&]() {
            decode_flash_p1_kernel<<<nb, DF_BLOCK>>>(d_q, d_K, d_V, d_pm, d_pd, d_pO, C);
            decode_flash_p2_kernel<<<1, D>>>(d_pm, d_pd, d_pO, d_o, nb);
        };
        fn(); CUDA_CHECK(cudaDeviceSynchronize());
        printf("  [verify flash:    %s]\n", verify_decode(d_o, h_q, h_K, h_V, C) ? "PASS" : "FAIL");
        bench_gpu("decode_flash (slice=256)", fn, flops);
        CUDA_CHECK(cudaFree(d_pm)); CUDA_CHECK(cudaFree(d_pd)); CUDA_CHECK(cudaFree(d_pO));
    }

    {
        int nb = (C + DF2_SLICE - 1) / DF2_SLICE;
        float *d_pm, *d_pd, *d_pO;
        CUDA_CHECK(cudaMalloc(&d_pm, nb * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pd, nb * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pO, (long long)nb * D * sizeof(float)));
        auto fn = [&]() {
            decode_flash_v2_p1_kernel<<<nb, D>>>(d_q, d_K, d_V, d_pm, d_pd, d_pO, C);
            decode_flash_p2_kernel<<<1, D>>>(d_pm, d_pd, d_pO, d_o, nb);
        };
        fn(); CUDA_CHECK(cudaDeviceSynchronize());
        printf("  [verify flash_v2: %s]\n", verify_decode(d_o, h_q, h_K, h_V, C) ? "PASS" : "FAIL");
        bench_gpu("decode_flash_v2 (slice=256, float4)", fn, flops);
        CUDA_CHECK(cudaFree(d_pm)); CUDA_CHECK(cudaFree(d_pd)); CUDA_CHECK(cudaFree(d_pO));
    }

    CUDA_CHECK(cudaFree(d_q)); CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_o));
    free(h_q); free(h_K); free(h_V);
}

// ============================================================
//  MAIN
// ============================================================

int main(int argc, char** argv)
{
    const char* mode = (argc > 1) ? argv[1] : "all";
    int         size = (argc > 2) ? atoi(argv[2]) : 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== Attention CUDA   device: %s   D=%d ===\n\n", prop.name, D);
    bool run_prefill = (strcmp(mode, "all") == 0 || strcmp(mode, "prefill") == 0);
    bool run_decode  = (strcmp(mode, "all") == 0 || strcmp(mode, "decode")  == 0);
    if (run_prefill) {
        if (size == 0 || size == 4096)  bench_prefill(4096);
        if (size == 0 || size == 65536) bench_prefill(65536);
    }
    if (run_decode) {
        if (size == 0 || size == 4096)  bench_decode(4096);
        if (size == 0 || size == 65536) bench_decode(65536);
    }
    printf("\nDone.\n");
    return 0;
}
