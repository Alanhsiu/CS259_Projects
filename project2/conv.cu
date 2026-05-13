/*
 * conv.cu  –  CS259 Mini-Project 1, Part 1: 2-D Convolution
 *
 * Usage:  ./conv [layer] [kernel_id]
 *   layer     : 1=Conv1, 2=Conv2, 0/omit=both
 *   kernel_id : 0=naive 1=smem_w 2=unroll 3=smem_wi 4=regblock 5=warp
 *               -1/omit = all kernels
 *
 * Layout (matching the course reference conv.cpp):
 *   synapse  [KY][KX][Nn][Ni]
 *   neuron_i [B*NYPAD][NXPAD][Ni]   (zero-padded via calloc)
 *   neuron_n [B*Ny][Nx][Nn]
 *
 * Kernels
 *   K0  conv_naive_kernel    – baseline: one thread owns one output element
 *   K1  conv_smem_kernel     – shared-mem weight tile (all threads share same nn)
 *   K2  conv_unroll_kernel   – K0 + #pragma unroll on ky/kx (compile-time loops)
 *   K3  conv_smem_wi_kernel  – shared-mem for both weight slice AND input patch
 *   K4  conv_regblock_kernel – register blocking: each thread computes REG_N outputs
 *   K5  conv_warp_kernel     – warp-level ni-reduction via __shfl_down_sync
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <cuda_runtime.h>

#include "conv_ref.h"   /* CPU reference, fill_*, max_abs_error */

/* ── error check ──────────────────────────────────────────────────────────── */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr,"CUDA error %s:%d  %s\n",                           \
                    __FILE__,__LINE__,cudaGetErrorString(_e));                  \
            exit(1);                                                            \
        }                                                                       \
    } while(0)

/* ── tile / block constants ──────────────────────────────────────────────── */
#define TILE_X       16   /* block x-dim for K0–K4                            */
#define TILE_Y       16   /* block y-dim for K0–K4                            */
#ifndef SMWI_TILE_NI
#define SMWI_TILE_NI  8   /* ni chunk size for K3 (smem_wi)                   */
#endif
#define REG_N         4   /* output channels per thread for K4 (regblock)     */
#define WARP_TILE_X   8   /* output-x positions per block for K5 (warp)       */

/* derived tile sizes for K3 */
#define INP_TILE_Y  (TILE_Y + KY - 1)   /* = 18 */
#define INP_TILE_X  (TILE_X + KX - 1)   /* = 18 */


/* ═══════════════════════════════════════════════════════════════════════════
 * K0 – Naive
 *
 * Grid : (⌈Nx/TILE_X⌉, ⌈B·Ny/TILE_Y⌉, Nn)    Block: (TILE_X, TILE_Y, 1)
 *
 * Each thread owns exactly one output element (b, yout, xout, nn) and
 * walks (ky, kx, ni) in the inner loops reading directly from DRAM every time.
 * ═══════════════════════════════════════════════════════════════════════════*/
__global__ void conv_naive_kernel(
    const VTYPE * __restrict__ syn,
    const VTYPE * __restrict__ inp,
    VTYPE       * __restrict__ out,
    int B, int Ny, int Nx, int Ni, int Nn)
{
    const int NYPAD = Ny + KY - 1, NXPAD = Nx + KX - 1;
    int xout = blockIdx.x * TILE_X + threadIdx.x;
    int by   = blockIdx.y * TILE_Y + threadIdx.y;   /* flat b*Ny + yout */
    int nn   = blockIdx.z;
    if (xout >= Nx || by >= B*Ny || nn >= Nn) return;
    int b = by/Ny, yout = by%Ny;

    VTYPE sum = 0.f;
    for (int ky = 0; ky < KY; ky++)
    for (int kx = 0; kx < KX; kx++)
    for (int ni = 0; ni < Ni; ni++)
        sum += syn[((ky*KX+kx)*Nn+nn)*Ni+ni] // syn[ky][kx][nn][ni]
             * inp[((long long)(b*NYPAD+ky+yout)*NXPAD+kx+xout)*Ni+ni]; // inp[b][y+ky][x+kx][ni]

    out[((long long)(b*Ny+yout)*Nx+xout)*Nn+nn] = fmaxf(0.f, sum);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * K1 – Shared-Memory Weight Tile
 *
 * All 256 threads in a block share blockIdx.z (= nn), so they all need the
 * SAME weight slice weight[*][*][nn][*].  Load it cooperatively into shared
 * memory once per block; subsequent accesses hit L1 (~5 cycles) not DRAM.
 *
 * smem = KY * KX * Ni * sizeof(float):
 *   Conv1 (Ni= 64):  2.25 KB  → thread-limited to 8 blk/SM → no penalty
 *   Conv2 (Ni=512): 18.00 KB  → smem-limited to 5 blk/SM  → occupancy drop
 * ═══════════════════════════════════════════════════════════════════════════*/
__global__ void conv_smem_kernel(
    const VTYPE * __restrict__ syn,
    const VTYPE * __restrict__ inp,
    VTYPE       * __restrict__ out,
    int B, int Ny, int Nx, int Ni, int Nn)
{
    extern __shared__ VTYPE smem_w[];   /* [KY * KX * Ni] */
    const int NYPAD = Ny+KY-1, NXPAD = Nx+KX-1;
    int xout = blockIdx.x*TILE_X + threadIdx.x;
    int by   = blockIdx.y*TILE_Y + threadIdx.y;
    int nn   = blockIdx.z;
    if (nn >= Nn) return;

    /* cooperative load */
    int tid     = threadIdx.y*TILE_X + threadIdx.x;
    int total_w = KY*KX*Ni;
    for (int t = tid; t < total_w; t += TILE_X*TILE_Y) {
        int ni = t%Ni, kyx = t/Ni, kx_ = kyx%KX, ky_ = kyx/KX;
        smem_w[t] = syn[((ky_*KX+kx_)*Nn+nn)*Ni+ni];
    }
    __syncthreads();

    if (xout >= Nx || by >= B*Ny) return;
    int b = by/Ny, yout = by%Ny;

    VTYPE sum = 0.f;
    for (int ky = 0; ky < KY; ky++)
    for (int kx = 0; kx < KX; kx++)
    for (int ni = 0; ni < Ni; ni++)
        sum += smem_w[(ky*KX+kx)*Ni+ni] // smem_w[ky][kx][ni]
             * inp[((long long)(b*NYPAD+ky+yout)*NXPAD+kx+xout)*Ni+ni]; // inp[b][y+ky][x+kx][ni]

    out[((long long)(b*Ny+yout)*Nx+xout)*Nn+nn] = fmaxf(0.f, sum);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * K2 – Loop Unrolling  (#pragma unroll)
 *
 * KY = KX = 3 are compile-time constants.  Unrolling exposes 9 independent
 * FMA chains to the scheduler, reducing loop-control overhead and enabling
 * better instruction-level parallelism.  The ni loop stays dynamic.
 * ═══════════════════════════════════════════════════════════════════════════*/
__global__ void conv_unroll_kernel(
    const VTYPE * __restrict__ syn,
    const VTYPE * __restrict__ inp,
    VTYPE       * __restrict__ out,
    int B, int Ny, int Nx, int Ni, int Nn)
{
    const int NYPAD = Ny+KY-1, NXPAD = Nx+KX-1;
    int xout = blockIdx.x*TILE_X + threadIdx.x;
    int by   = blockIdx.y*TILE_Y + threadIdx.y;
    int nn   = blockIdx.z;
    if (xout >= Nx || by >= B*Ny || nn >= Nn) return;
    int b = by/Ny, yout = by%Ny;

    VTYPE sum = 0.f;
    #pragma unroll
    for (int ky = 0; ky < KY; ky++)
    #pragma unroll
    for (int kx = 0; kx < KX; kx++)
    for (int ni = 0; ni < Ni; ni++)
        sum += syn[((ky*KX+kx)*Nn+nn)*Ni+ni]
             * inp[((long long)(b*NYPAD+ky+yout)*NXPAD+kx+xout)*Ni+ni];

    out[((long long)(b*Ny+yout)*Nx+xout)*Nn+nn] = fmaxf(0.f, sum);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * K3 – Shared-Memory Weight + Input Patch  (smem_wi)
 *
 * Extends K1 by also tiling the INPUT into shared memory, so both weight and
 * input accesses hit the fast on-chip SRAM instead of DRAM / L2.
 *
 * Grid restructured so every thread in a block has the SAME (b, nn):
 *   Grid : (⌈Nx/TILE_X⌉,  ⌈Ny/TILE_Y⌉,  B*Nn)
 *   blockIdx.z = b * Nn + nn    →  b = blockIdx.z/Nn,  nn = blockIdx.z%Nn
 * This prevents batch-boundary crossing inside a block, making the input tile
 * layout contiguous and well-defined.
 *
 * For each SMWI_TILE_NI-wide ni chunk:
 *   1. Load  weight[ky][kx][nn][ni_chunk]  into smem_w  (2.25 KB per chunk)
 *   2. Load  input patch [18][18][ni_chunk] into smem_i (10.1 KB per chunk)
 *   3. Compute using smem; repeat.
 *
 * Total smem = (9*8 + 18*18*8)*4 = 10,656 bytes ≈ 10.4 KB/block
 *   96 KB / 10.4 KB ≈ 9 blocks/SM, but thread-limited to 8 → no penalty.
 * ═══════════════════════════════════════════════════════════════════════════*/
__global__ void conv_smem_wi_kernel(
    const VTYPE * __restrict__ syn,
    const VTYPE * __restrict__ inp,
    VTYPE       * __restrict__ out,
    int B, int Ny, int Nx, int Ni, int Nn)
{
    /*
     * smem layout:  [ smem_w | smem_i ]
     *   smem_w : KY * KX * SMWI_TILE_NI  floats
     *   smem_i : INP_TILE_Y * INP_TILE_X * SMWI_TILE_NI  floats
     */
    extern __shared__ VTYPE smem[];
    VTYPE *smem_w = smem;
    VTYPE *smem_i = smem + KY*KX*SMWI_TILE_NI;

    const int NYPAD = Ny+KY-1, NXPAD = Nx+KX-1;

    int nn        = blockIdx.z % Nn;
    int b         = blockIdx.z / Nn;
    int yout_base = blockIdx.y * TILE_Y;
    int xout_base = blockIdx.x * TILE_X;
    int yout      = yout_base + threadIdx.y;
    int xout      = xout_base + threadIdx.x;
    int tid       = threadIdx.y*TILE_X + threadIdx.x;   /* 0..255 */

    VTYPE sum = 0.f;

    for (int ni0 = 0; ni0 < Ni; ni0 += SMWI_TILE_NI) {
        int ni_count = (ni0 + SMWI_TILE_NI <= Ni) ? SMWI_TILE_NI : (Ni - ni0);

        /* 1. load weight slice */
        for (int t = tid; t < KY*KX*SMWI_TILE_NI; t += TILE_X*TILE_Y) {
            int ni_off = t % SMWI_TILE_NI;
            int kyx    = t / SMWI_TILE_NI;
            int kx_    = kyx%KX, ky_ = kyx/KX;
            int ni_abs = ni0 + ni_off;
            smem_w[t] = (ni_abs < Ni) ? syn[((ky_*KX+kx_)*Nn+nn)*Ni+ni_abs] : 0.f;
        }

        /* 2. load input patch  [INP_TILE_Y][INP_TILE_X][SMWI_TILE_NI] */
        for (int t = tid; t < INP_TILE_Y*INP_TILE_X*SMWI_TILE_NI; t += TILE_X*TILE_Y) {
            int ni_off = t % SMWI_TILE_NI;
            int rem    = t / SMWI_TILE_NI;
            int lx     = rem % INP_TILE_X;
            int ly     = rem / INP_TILE_X;
            int ni_abs = ni0 + ni_off;
            bool ok    = (yout_base+ly < NYPAD) && (xout_base+lx < NXPAD) && (ni_abs < Ni);
            smem_i[t]  = ok ? inp[((long long)(b*NYPAD+yout_base+ly)*NXPAD
                                   +xout_base+lx)*Ni+ni_abs] : 0.f;
        }

        __syncthreads();

        /* 3. compute: thread (threadIdx.y, threadIdx.x) → output (yout, xout) */
        if (yout < Ny && xout < Nx) {
            for (int ky = 0; ky < KY; ky++)
            for (int kx = 0; kx < KX; kx++)
            for (int ni_off = 0; ni_off < ni_count; ni_off++) {
                int ly = threadIdx.y + ky;
                int lx = threadIdx.x + kx;
                sum += smem_w[(ky*KX+kx)*SMWI_TILE_NI + ni_off]
                     * smem_i[(ly*INP_TILE_X+lx)*SMWI_TILE_NI + ni_off];
            }
        }

        __syncthreads();
    }

    if (yout < Ny && xout < Nx)
        out[((long long)(b*Ny+yout)*Nx+xout)*Nn+nn] = fmaxf(0.f, sum);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * K4 – Register Blocking
 *
 * Each thread computes REG_N = 4 consecutive output channels (nn_base..+3).
 * The input value inp[b][y+ky][x+kx][ni] is loaded ONCE and reused for
 * REG_N multiply-accumulates, amortising input DRAM traffic by ~4×.
 *
 * Grid : (⌈Nx/TILE_X⌉, ⌈B·Ny/TILE_Y⌉, ⌈Nn/REG_N⌉)
 * Block: (TILE_X, TILE_Y, 1)  — same as K0
 * ═══════════════════════════════════════════════════════════════════════════*/
__global__ void conv_regblock_kernel(
    const VTYPE * __restrict__ syn,
    const VTYPE * __restrict__ inp,
    VTYPE       * __restrict__ out,
    int B, int Ny, int Nx, int Ni, int Nn)
{
    const int NYPAD = Ny+KY-1, NXPAD = Nx+KX-1;
    int xout    = blockIdx.x*TILE_X + threadIdx.x;
    int by      = blockIdx.y*TILE_Y + threadIdx.y;
    int nn_base = blockIdx.z * REG_N;
    if (xout >= Nx || by >= B*Ny || nn_base >= Nn) return;
    int b = by/Ny, yout = by%Ny;

    VTYPE sum[REG_N] = {};

    for (int ky = 0; ky < KY; ky++)
    for (int kx = 0; kx < KX; kx++)
    for (int ni = 0; ni < Ni; ni++) {
        VTYPE in_val = inp[((long long)(b*NYPAD+ky+yout)*NXPAD+kx+xout)*Ni+ni];
        #pragma unroll
        for (int r = 0; r < REG_N; r++) {
            int nn = nn_base + r;
            if (nn < Nn)
                sum[r] += syn[((ky*KX+kx)*Nn+nn)*Ni+ni] * in_val;
        }
    }

    #pragma unroll
    for (int r = 0; r < REG_N; r++) {
        int nn = nn_base + r;
        if (nn < Nn)
            out[((long long)(b*Ny+yout)*Nx+xout)*Nn+nn] = fmaxf(0.f, sum[r]);
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
 * K5 – Warp-Level ni Reduction  (__shfl_down_sync)
 *
 * One full WARP (32 lanes) collaborates on a single output element's dot
 * product over (ky, kx, ni).  Lane k handles the subset of the flat iteration
 *     t = ky*KX*Ni + kx*Ni + ni
 * where t ≡ k (mod 32).
 *
 * Coalescing benefit:
 *   Within a warp (same ky, kx), adjacent lanes access ni = 0,1,2,… in
 *   syn[...][ni] and inp[...][ni] → stride 1 → perfectly coalesced reads.
 *   This is better than K0, where adjacent threads have different xout,
 *   producing stride-Ni accesses in the input (256 bytes for Conv1).
 *
 * After partial sums are accumulated, __shfl_down_sync butterfly-reduces
 * all 32 lanes into lane 0, which writes the output (no shared memory needed).
 *
 * Block : (32, WARP_TILE_X, 1) = 256 threads = 8 warps
 *   threadIdx.x = warp lane (0..31)       → ni partition
 *   threadIdx.y = x-position slot (0..7)  → distinct xout per warp
 * Grid  : (⌈Nx/WARP_TILE_X⌉, B*Ny, Nn)
 * ═══════════════════════════════════════════════════════════════════════════*/
__global__ void conv_warp_kernel(
    const VTYPE * __restrict__ syn,
    const VTYPE * __restrict__ inp,
    VTYPE       * __restrict__ out,
    int B, int Ny, int Nx, int Ni, int Nn)
{
    const int NYPAD = Ny+KY-1, NXPAD = Nx+KX-1;
    int xout = blockIdx.x*WARP_TILE_X + threadIdx.y;   /* unique per warp in block */
    int by   = blockIdx.y;                              /* flat b*Ny + yout */
    int nn   = blockIdx.z;
    int lane = threadIdx.x;                             /* 0..31 */

    if (xout >= Nx || by >= B*Ny || nn >= Nn) return;
    int b = by/Ny, yout = by%Ny;

    VTYPE part = 0.f;
    int total = KY*KX*Ni;
    for (int t = lane; t < total; t += 32) {
        int ni  = t % Ni;
        int kyx = t / Ni;
        int kx_ = kyx%KX, ky_ = kyx/KX;
        part += syn[((ky_*KX+kx_)*Nn+nn)*Ni+ni]
              * inp[((long long)(b*NYPAD+ky_+yout)*NXPAD+kx_+xout)*Ni+ni];
    }

    /* butterfly reduce: lane 0 accumulates the full sum */
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        part += __shfl_down_sync(0xffffffff, part, off);

    if (lane == 0)
        out[((long long)(b*Ny+yout)*Nx+xout)*Nn+nn] = fmaxf(0.f, part);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * Timing helper
 * ═══════════════════════════════════════════════════════════════════════════*/
#define NUM_RUNS 5

template <typename F>
static float time_kernel_ms(F launch)
{
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    float ms[NUM_RUNS];
    for (int r = 0; r < NUM_RUNS; r++) {
        CUDA_CHECK(cudaEventRecord(t0));
        launch();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms[r], t0, t1));
    }
    std::sort(ms, ms+NUM_RUNS);
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms[NUM_RUNS/2];
}


/* ═══════════════════════════════════════════════════════════════════════════
 * run_config
 * ═══════════════════════════════════════════════════════════════════════════*/
static void run_config(const char *name,
                       int B, int Ny, int Nx, int Ni, int Nn,
                       unsigned kernel_mask)
{
    const int  NYPAD = Ny+KY-1, NXPAD = Nx+KX-1;
    const long long syn_n = (long long)KY*KX*Nn*Ni;
    const long long inp_n = (long long)B*NYPAD*NXPAD*Ni;
    const long long out_n = (long long)B*Ny*Nx*Nn;

    VTYPE *h_syn = (VTYPE*)malloc(syn_n*sizeof(VTYPE));
    VTYPE *h_inp = (VTYPE*)calloc(inp_n,  sizeof(VTYPE));
    VTYPE *h_cpu = (VTYPE*)malloc(out_n*sizeof(VTYPE));
    VTYPE *h_gpu = (VTYPE*)malloc(out_n*sizeof(VTYPE));

    fill_synapse(h_syn, syn_n);
    fill_input(h_inp, B, Ny, Nx, Ni);

    printf("\n=== %s  (B=%d, Ny=%d, Nx=%d, Ni=%d, Nn=%d) ===\n",
           name, B, Ny, Nx, Ni, Nn);
    printf("  Running CPU reference"); fflush(stdout);
    conv_cpu_reference(h_syn, h_inp, h_cpu, B, Ny, Nx, Ni, Nn);
    printf(" ... done.\n\n");
    printf("  %-14s  %10s  %10s  %12s  %6s\n",
           "Kernel","Time(ms)","GFLOPS","MaxAbsErr","Status");
    printf("  %s\n",
           "──────────────────────────────────────────────────────────────");

    VTYPE *d_syn, *d_inp, *d_out;
    CUDA_CHECK(cudaMalloc(&d_syn, syn_n*sizeof(VTYPE)));
    CUDA_CHECK(cudaMalloc(&d_inp, inp_n*sizeof(VTYPE)));
    CUDA_CHECK(cudaMalloc(&d_out, out_n*sizeof(VTYPE)));
    CUDA_CHECK(cudaMemcpy(d_syn, h_syn, syn_n*sizeof(VTYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inp, h_inp, inp_n*sizeof(VTYPE), cudaMemcpyHostToDevice));

    const long long flops = 2LL*B*Ny*Nx*Nn*KY*KX*Ni;

    auto report = [&](const char *kname, float ms_med) {
        CUDA_CHECK(cudaMemcpy(h_gpu, d_out, out_n*sizeof(VTYPE), cudaMemcpyDeviceToHost));
        float err    = max_abs_error(h_gpu, h_cpu, out_n);
        float gflops = (float)flops*1e-9f / (ms_med*1e-3f);
        printf("  %-14s  %10.3f  %10.2f  %12.2e  [%s]\n",
               kname, ms_med, gflops, err, err<1e-3f?"PASS":"FAIL");
    };

    /* common grid for K0,K1,K2 */
    dim3 blk(TILE_X, TILE_Y, 1);
    dim3 grid0( (Nx+TILE_X-1)/TILE_X, (B*Ny+TILE_Y-1)/TILE_Y, Nn );

    /* K0 – naive */
    if (kernel_mask & (1u<<0)) {
        CUDA_CHECK(cudaMemset(d_out, 0, out_n*sizeof(VTYPE)));
        float ms = time_kernel_ms([&](){
            conv_naive_kernel<<<grid0,blk>>>(d_syn,d_inp,d_out,B,Ny,Nx,Ni,Nn);
        });
        report("naive", ms);
    }

    /* K1 – smem_w */
    if (kernel_mask & (1u<<1)) {
        size_t sm1 = (size_t)KY*KX*Ni*sizeof(VTYPE);
        CUDA_CHECK(cudaMemset(d_out, 0, out_n*sizeof(VTYPE)));
        float ms = time_kernel_ms([&](){
            conv_smem_kernel<<<grid0,blk,sm1>>>(d_syn,d_inp,d_out,B,Ny,Nx,Ni,Nn);
        });
        report("smem_w", ms);
    }

    /* K2 – unroll */
    if (kernel_mask & (1u<<2)) {
        CUDA_CHECK(cudaMemset(d_out, 0, out_n*sizeof(VTYPE)));
        float ms = time_kernel_ms([&](){
            conv_unroll_kernel<<<grid0,blk>>>(d_syn,d_inp,d_out,B,Ny,Nx,Ni,Nn);
        });
        report("unroll", ms);
    }

    /* K3 – smem_wi  (restructured grid: blockIdx.z = b * Nn + nn) */
    if (kernel_mask & (1u<<3)) {
        dim3 grid3( (Nx+TILE_X-1)/TILE_X, (Ny+TILE_Y-1)/TILE_Y, B*Nn );
        size_t sm3 = ((size_t)KY*KX*SMWI_TILE_NI
                    + (size_t)INP_TILE_Y*INP_TILE_X*SMWI_TILE_NI) * sizeof(VTYPE);
        CUDA_CHECK(cudaMemset(d_out, 0, out_n*sizeof(VTYPE)));
        float ms = time_kernel_ms([&](){
            conv_smem_wi_kernel<<<grid3,blk,sm3>>>(d_syn,d_inp,d_out,B,Ny,Nx,Ni,Nn);
        });
        report("smem_wi", ms);
    }

    /* K4 – regblock */
    if (kernel_mask & (1u<<4)) {
        dim3 grid4( (Nx+TILE_X-1)/TILE_X, (B*Ny+TILE_Y-1)/TILE_Y,
                    (Nn+REG_N-1)/REG_N );
        CUDA_CHECK(cudaMemset(d_out, 0, out_n*sizeof(VTYPE)));
        float ms = time_kernel_ms([&](){
            conv_regblock_kernel<<<grid4,blk>>>(d_syn,d_inp,d_out,B,Ny,Nx,Ni,Nn);
        });
        report("regblock", ms);
    }

    /* K5 – warp */
    if (kernel_mask & (1u<<5)) {
        dim3 blk5(32, WARP_TILE_X, 1);
        dim3 grid5( (Nx+WARP_TILE_X-1)/WARP_TILE_X, B*Ny, Nn );
        CUDA_CHECK(cudaMemset(d_out, 0, out_n*sizeof(VTYPE)));
        float ms = time_kernel_ms([&](){
            conv_warp_kernel<<<grid5,blk5>>>(d_syn,d_inp,d_out,B,Ny,Nx,Ni,Nn);
        });
        report("warp", ms);
    }

    cudaFree(d_syn); cudaFree(d_inp); cudaFree(d_out);
    free(h_syn); free(h_inp); free(h_cpu); free(h_gpu);
}


/* ═══════════════════════════════════════════════════════════════════════════
 * main
 * ═══════════════════════════════════════════════════════════════════════════*/
int main(int argc, char *argv[])
{
    int layer_sel  = (argc > 1) ? atoi(argv[1]) : 0;
    int kernel_sel = (argc > 2) ? atoi(argv[2]) : -1;
    int Ny         = (argc > 3) ? atoi(argv[3]) : 0;  /* spatial size (0 = use default) */
    int Nx         = (argc > 4) ? atoi(argv[4]) : Ny; /* if omitted, Nx = Ny */
    unsigned kmask = (kernel_sel < 0) ? 0xFFFFFFFFu : (1u << kernel_sel);

    {
        cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
        printf("Device : %s  (CC %d.%d)\n", p.name, p.major, p.minor);
    }

    if (Ny == 0) {
        /* Default sweeps */
        if (layer_sel != 2) run_config("Conv1-VGG", 16, 224, 224,  64,  64, kmask);
        if (layer_sel != 1) run_config("Conv2-VGG", 16,  14,  14, 512, 512, kmask);
    } else {
        /* Custom spatial sizes for sweep */
        run_config("Conv1-Custom", 16, Ny, Nx,  64,  64, kmask);
    }
    printf("\n");
    return 0;
}
