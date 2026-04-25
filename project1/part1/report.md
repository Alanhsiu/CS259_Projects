# CS259 Mini-Project 1 — Part 1: Convolution

**GPU:** NVIDIA TITAN V (Volta GV100, CC 7.0)  
**Batch size:** B = 16 (used consistently for all configurations)  
**Block size:** 16 × 16 × 1 = 256 threads/block  

---

## Q1. Parallelization Strategy

The four output dimensions `[B][Ny][Nx][Nn]` are mapped to the CUDA grid as follows:

| Output dimension | Mapped to | Rationale |
|---|---|---|
| `Nn` (output channel) | `blockIdx.z` | Each block exclusively owns one output channel; weight slice `weight[*][*][nn][*]` is fixed per block, enabling shared-memory reuse (smem_w kernel) |
| `B × Ny` (batch × row, flattened) | `blockIdx.y × TILE_Y + threadIdx.y` | Batch and spatial-y are independent, so they are collapsed into one grid axis |
| `Nx` (spatial x) | `blockIdx.x × TILE_X + threadIdx.x` | Adjacent threads write adjacent `Nn`-minor output addresses → coalesced writes |

The inner loop of each thread iterates over `(ky, kx, ni)` to accumulate the dot product for one output element.

**Grid configuration (Conv1, B=16):**
```
dim3 block(16, 16, 1);          // 256 threads/block
dim3 grid(14, 224, 64);         // (⌈224/16⌉, ⌈16×224/16⌉, 64)
```

**Limitations:**

- *Channel count Nn:* `blockIdx.z = nn` maps each block to exactly one output channel. This is natural for small Nn (64), but for Nn = 512 the grid Z-dimension grows to 512, potentially limiting SM utilisation if occupancy is already constrained by other resources.
- *Inner-loop serialisation:* The `ni` loop (Ni = 64 or 512) runs serially inside each thread. For Conv2 (Ni = 512) this is 9 × 512 = 4,608 iterations per thread — compute is serial within a thread, so inter-SM parallelism fully determines throughput.
- *Scaling with context:* The strategy scales linearly with B, Ny, Nx, and Nn; it does not scale the inner Ni dimension in parallel, which is an opportunity for future register-blocking or warp-level reduction optimisations.

---

## Q2. Algorithmic FLOP Count

Each output element requires `Ky × Kx × Ni` multiply-add operations (one FMA = 2 FLOPs):

$$\text{FLOPs} = 2 \times B \times N_y \times N_x \times N_n \times K_y \times K_x \times N_i$$

### Conv1 (B=16, 224×224, Ni=Nn=64, Ky=Kx=3)

$$= 2 \times 16 \times 224 \times 224 \times 64 \times 3 \times 3 \times 64 = \mathbf{59.19 \text{ GFLOPs}}$$

*Verified by ncu:* `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` = 29,595,009,024 → 29,595,009,024 × 2 = 59,190,018,048 FLOPs ✓

### Conv2 (B=16, 14×14, Ni=Nn=512, Ky=Kx=3)

$$= 2 \times 16 \times 14 \times 14 \times 512 \times 3 \times 3 \times 512 = \mathbf{14.80 \text{ GFLOPs}}$$

*Verified by ncu:* 7,398,752,256 × 2 = 14,797,504,512 FLOPs ✓

---

## Q3. Execution Time and Achieved GFLOPS

Times are median of 5 runs measured with CUDA Events, without profiler overhead.

| Config | Kernel | Time (ms) | Achieved GFLOPS | Max Abs Err | Status |
|--------|--------|-----------|-----------------|-------------|--------|
| Conv1 (224×224, Ni=Nn=64) | naive | 685.77 | 86.31 | 1.16 × 10⁻⁹ | PASS |
| Conv1 | smem\_w | 664.69 | 89.05 | 1.16 × 10⁻⁹ | PASS |
| Conv2 (14×14, Ni=Nn=512) | naive | 84.87 | 174.35 | 6.28 × 10⁻¹⁰ | PASS |
| Conv2 | smem\_w | 99.75 | 148.34 | 6.28 × 10⁻¹⁰ | PASS |

**TITAN V peak FP32:** 14,900 GFLOPS — so Conv1 achieves ~0.58% of peak and Conv2 ~1.17% of peak. Both are far from the hardware ceiling, indicating memory bottlenecks.

---

## Q4. Roofline Analysis

### Hardware parameters — NVIDIA TITAN V

| Metric | Value |
|--------|-------|
| Peak FP32 | 14,900 GFLOPS (14.9 TFLOPS) |
| Peak HBM2 bandwidth | 652.8 GB/s |
| **Ridge point** | **22.83 FLOPs/byte** |
| L2 cache | 4.5 MB |
| Shared memory/SM | 96 KB |

> **Unit note:** ncu reports `dram__bytes_*.sum` in `Gbyte` = 10⁹ bytes (SI). This is confirmed by: achieved BW = DRAM bytes / kernel time matches `gpu__dram_throughput` percentage within 0.5%.

---

### Theoretical Arithmetic Intensity

Minimum bytes = read each tensor exactly once (synapse + neuron\_i + neuron\_n).

**Conv1:**

| Tensor | Size |
|--------|------|
| synapse `[3][3][64][64]` | 9 × 64 × 64 × 4 = 147,456 B (0.14 MB) |
| neuron\_i `[16×226][226][64]` | 16 × 226 × 226 × 64 × 4 = 208,949,248 B (199.3 MB) |
| neuron\_n `[16×224][224][64]` | 16 × 224 × 224 × 64 × 4 = 205,520,896 B (196.0 MB) |
| **Total minimum** | **414,617,600 B (395.5 MB)** |

$$\text{AI}_{\text{theory,Conv1}} = \frac{59.19 \times 10^9}{414.6 \times 10^6} \approx \mathbf{142.8 \text{ FLOPs/byte}}$$

**Conv2:**

| Tensor | Size |
|--------|------|
| synapse `[3][3][512][512]` | 9,437,184 B (9.0 MB) |
| neuron\_i `[16×16][16][512]` | 8,388,608 B (8.0 MB) |
| neuron\_n `[16×14][14][512]` | 6,422,528 B (6.1 MB) |
| **Total minimum** | **24,248,320 B (23.1 MB)** |

$$\text{AI}_{\text{theory,Conv2}} = \frac{14.80 \times 10^9}{24.25 \times 10^6} \approx \mathbf{610.2 \text{ FLOPs/byte}}$$

Both theoretical AI values exceed the ridge point (22.83), so *both configurations should be compute-bound in the ideal case* — if every byte fetched from DRAM is maximally reused.

---

### Measured Arithmetic Intensity (from ncu)

$$\text{AI}_{\text{measured}} = \frac{\text{FMA count} \times 2}{\texttt{dram\_\_bytes\_read.sum} + \texttt{dram\_\_bytes\_write.sum}}$$

| Config | Kernel | DRAM Read (GB) | DRAM Write (GB) | Total DRAM (GB) | Actual AI (FLOPs/B) | Bound |
|--------|--------|---------------|----------------|----------------|---------------------|-------|
| Conv1 | naive | 418.98 | 1.65 | 420.63 | **0.141** | Memory |
| Conv1 | smem\_w | 405.90 | 1.65 | 407.55 | **0.145** | Memory |
| Conv2 | naive | 14.06 | 0.043 | 14.10 | **1.049** | Memory |
| Conv2 | smem\_w | 17.59 | 0.047 | 17.64 | **0.839** | Memory |

All four kernels have measured AI ≪ 22.83 (ridge point), confirming all are **memory-bound** in practice.

---

### Roofline Chart

![Roofline – TITAN V, Conv1 and Conv2](results/roofline.png)

*Filled circles = naive; hollow squares = smem\_w. Dashed vertical lines show theoretical AI.*

---

### Gap Between Theoretical and Measured AI

| Config | Theory AI | Measured AI | DRAM excess factor |
|--------|-----------|-------------|-------------------|
| Conv1 naive | 142.8 | 0.141 | **~1013×** |
| Conv1 smem\_w | 142.8 | 0.145 | ~985× |
| Conv2 naive | 610.2 | 1.049 | **~582×** |
| Conv2 smem\_w | 610.2 | 0.839 | ~727× |

**Conv1:** The theoretical minimum requires reading the 144 KB weight tensor once and the 199 MB padded input once. In the naive kernel, every one of the 16 × 224 × 224 × 64 ≈ 51.4 M output elements independently re-fetches 576 weight values and 576 input values from DRAM/L2 with little reuse. This inflates actual DRAM traffic from ~0.41 GB to 420.63 GB — a ×1013 overhead. Even though the 144 KB weight fits in TITAN V's 4.5 MB L2 cache conceptually, the concurrent pressure from 51.4 M threads evicts weight cache lines before they can be reused, causing most weight reads to reach DRAM.

**Conv2:** The padded input is only 8 MB and the output 6 MB — both fit comfortably in L2 for a single pass. With only 16 × 14 × 14 = 3,136 spatial output positions per channel block, each weight element is accessed far fewer times than in Conv1, giving a lower DRAM excess factor (~582×). The L2 cache provides significant relief: `l1tex__t_bytes_pipe_lsu_mem_global_op_ld` = 244.97 GB while `dram__bytes_read` = only 14.06 GB, meaning ~94% of L1 misses are served by L2. Nevertheless, measured AI (1.049) is still far below the ridge point (22.83), placing Conv2 solidly in the memory-bound regime.

---

## Q5. Optimisations

### Optimisation 1: Shared-Memory Weight Tiling (`smem_w`)

**Idea.** All threads within a block share the same `blockIdx.z` (output channel `nn`), so they all need the identical weight slice `weight[*][*][nn][0..Ni-1]`. We load this slice cooperatively into shared memory before the computation phase, so subsequent weight accesses hit shared memory (< 1 cycle) rather than DRAM/L2 (hundreds of cycles).

Shared memory per block = Ky × Kx × Ni × 4 bytes:
- Conv1: 9 × 64 × 4 = **2.25 KB** (negligible; does not reduce occupancy)
- Conv2: 9 × 512 × 4 = **18.0 KB** (reduces max blocks/SM from 8 → 5)

---

**Conv1 results:**

| Metric | naive | smem\_w | Δ |
|--------|-------|---------|---|
| Time (ms) | 685.77 | 664.69 | **−3.1%** |
| GFLOPS | 86.31 | 89.05 | +3.2% |
| DRAM traffic (GB) | 420.63 | 407.55 | −3.1% |
| Actual AI (FLOPs/B) | 0.141 | 0.145 | +3.1% |
| DRAM throughput | 93.85% | 94.06% | ≈ same |
| SM throughput | 2.82% | 2.92% | ≈ same |
| Occupancy | 73.24% | 73.23% | ≈ same |

The improvement is real but very small. Because the weight tile is only 2.25 KB, it does not limit occupancy (max 8 blocks/SM both before and after). The DRAM traffic falls by 13 GB because weight reads that previously escaped to DRAM are now served from shared memory. However, since the kernel is still bottlenecked by the **input** reads (208 MB padded input accessed ~51 M times without spatial reuse), DRAM throughput barely changes (93.85% → 94.06%). Smem_w addresses the wrong bottleneck for Conv1: the weight was already partially cached in L2; what hurts more is the input.

---

**Conv2 results:**

| Metric | naive | smem\_w | Δ |
|--------|-------|---------|---|
| Time (ms) | 84.87 | 99.75 | **+17.5% slower** |
| GFLOPS | 174.35 | 148.34 | −14.9% |
| DRAM traffic (GB) | 14.10 | 17.64 | **+25.1%** |
| Actual AI (FLOPs/B) | 1.049 | 0.839 | −20.0% |
| DRAM throughput | 25.66% | 26.96% | ≈ same |
| SM throughput | 6.50% | 5.52% | −15.1% |
| Occupancy | 73.81% | 61.49% | **−16.7%** |

The smem_w kernel is **15% slower** for Conv2. The 18 KB shared memory per block reduces the maximum concurrent blocks per SM from 8 to 5 (96 KB ÷ 18 KB), dropping occupancy from 73.8% to 61.5%. With fewer resident warps, the SM has less ability to hide memory latency by switching to another warp while one is waiting — directly causing the 15% GFLOPS degradation.

Counter-intuitively, DRAM traffic *increases* by 25%: the reduced warp parallelism leaves the DRAM scheduler with fewer in-flight requests to batch, degrading cache-line utilisation and causing more L2 evictions. The L1→L2 miss volume confirms this (`lts__t_bytes_equiv_l1sectormiss`: 37.31 GB → 87.23 GB, a +133% increase), meaning many more L2 misses reach DRAM.

For Conv2 the weight is 9.4 MB — too large for shared memory to be loaded as one block-wide tile without hurting occupancy. A better strategy would be tiling the **input** into shared memory (the padded input is only 8 MB for the full batch, so a per-block input tile is small) or using register blocking to compute multiple output positions per thread to amortise both the weight and input loads.

---

### Summary

| Optimisation | Conv1 | Conv2 | Root cause |
|---|---|---|---|
| smem\_w | ✅ +3% | ❌ −15% | Weight tile fits in smem without occupancy cost for Conv1 (Ni=64, 2.25 KB), but hurts Conv2 (Ni=512, 18 KB reduces occupancy 16%) |

**What helped:** Shared memory weight tiling reduced DRAM traffic slightly for Conv1 (weight freed from L2 pressure).

**What did not help:** The same kernel worsened Conv2 by trading occupancy for a data-locality benefit that was already largely covered by L2 cache.

**Potential further optimisations:**
- **Input tiling (smem input strip):** Load a spatial strip of the padded input into shared memory so adjacent output positions can share input reads. This would reduce the dominant bottleneck for Conv1.
- **Register blocking:** Each thread computes REG_BLOCK output channels simultaneously, amortising the cost of loading input values across multiple output channels.
- **`#pragma unroll` on inner loops:** Especially the `ky`/`kx` loops (bounded at 3), allowing the compiler to remove loop overhead and improve instruction-level parallelism.
- **Warp-level primitives:** For a reduction-style variant where threads in a warp collaborate to compute a single output, `__shfl_sync` could replace shared-memory reductions.

---

*Source files:* `conv.cu`, `conv_ref.h` | *Profiling:* `run_ncu.sh` | *Plot:* `plot_roofline.py`
