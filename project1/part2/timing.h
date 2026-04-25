#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <functional>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(err) do {                                              \
    cudaError_t _e = (err);                                               \
    if (_e != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error at %s:%d  %s\n",                     \
                __FILE__, __LINE__, cudaGetErrorString(_e));              \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
} while (0)

static void bench_gpu(const char* name,
                      std::function<void()> fn,
                      double flops,
                      int warmup = 3,
                      int iters  = 10)
{
    for (int i = 0; i < warmup; i++) {
        fn();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    std::vector<float> ms;
    ms.reserve(iters);
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(t0));
        fn();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float elapsed;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, t0, t1));
        ms.push_back(elapsed);
    }
    std::sort(ms.begin(), ms.end());
    float med = ms[iters / 2];
    double gflops = flops / (med * 1e-3) / 1e9;
    printf("  %-48s  %8.3f ms  %8.2f GFLOPS\n", name, med, gflops);

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
}
