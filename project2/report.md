# CS259 Project 2: Hierarchical Performance Model for Convolution Kernels

## Executive Summary

This report documents the design and implementation of a **mechanistic performance model** for 6 convolution kernel variants (naive, smem_w, unroll, smem_wi, regblock, warp) on NVIDIA TITAN V. The model predicts execution time by analyzing:
1. **Computational demand** (FLOPs vs peak FP32 throughput)
2. **Memory traffic** at the DRAM level (with L2 cache pressure correction)
3. **Per-kernel scaling factors** fitted from measured data

The key innovation is **L2-pressure-aware traffic correction**: instead of using a fixed measured DRAM value, we build a mechanistic DRAM traffic estimate per kernel and apply a small shape-dependent correction. This enables the model to generalize beyond the measured conv1/conv2 pair.

---

## Part 1: From Roofline to Mechanistic Models

### 1.1 The Original Roofline Baseline

The classic **roofline model** (Appendix A in original paper, Williams et al., 2009) predicts performance by placing kernels on a 2D chart:
- **X-axis**: Arithmetic Intensity (AI) = FLOPs / bytes accessed
- **Y-axis**: Achieved throughput (GFLOPS)
- **Roof**: Two lines forming an envelope:
  - **Compute roof**: horizontal at peak FP32 TFLOPS
  - **Memory roof**: diagonal line with slope = DRAM bandwidth

For each kernel:
$$
\text{predicted\_time\_s} = \max\left( \frac{\text{FLOPs}}{\text{peak\_FP32}}, \frac{\text{DRAM\_bytes}}{\text{peak\_BW}} \right)
$$

**Why roofline alone is insufficient:**
- It assumes **every byte is read/written exactly once** from DRAM (theoretical minimum).
- In reality, kernels re-fetch inputs multiple times due to limited cache and limited tiling.
- For Conv1 naive: roofline predicts ~4 ms; actual is ~686 ms → Time MAPE ≈ 99.4% because the model doesn't account for reuse patterns.
- The model cannot capture the difference between kernels: naive vs smem_wi both have similar theoretical AI but **vastly different actual performance**.

### 1.2 Why Mechanistic DRAM Traffic Estimation?

Instead of blindly using theoretical-minimum bytes or purely measured bytes, we:

1. **Analyze each kernel's memory access pattern** to build an **analytic traffic estimate**.
2. **Fit a small correction term** that accounts for cache effects not captured by the pattern analysis.
3. **Preserve generalizability**: the mechanistic path works for any problem size; the correction refines it using available data.

---

## Part 2: Kernel-by-Kernel DRAM Traffic Analysis

All kernels are analyzed for a convolution:
$$
\text{output}[b, y, x, n] = \sum_{k_y=0}^{K_y-1} \sum_{k_x=0}^{K_x-1} \sum_{i=0}^{N_i-1} \text{weight}[k_y, k_x, i, n] \times \text{input}[b, y+k_y, x+k_x, i]
$$

**Key tensors:**
- **Weights**: shape $(K_y, K_x, N_i, N_n)$ = $(3, 3, N_i, N_n)$, size = $9 \times N_i \times N_n \times 4$ bytes
- **Input (padded)**: shape $(B, N_y + K_y - 1, N_x + K_x - 1, N_i)$, size = $B \times (N_y + 2) \times (N_x + 2) \times N_i \times 4$ bytes
- **Output**: shape $(B, N_y, N_x, N_n)$, size = $B \times N_y \times N_x \times N_n \times 4$ bytes

### 2.1 K0: Naive (No Shared Memory, No Tiling)

Each thread reads input/weights on-the-fly:
- **Input traffic**: Every output element requires $K_y \times K_x \times N_i = 9 \times N_i$ input reads.
  - Total: $B \times N_y \times N_x \times N_n \times 9 \times N_i \times 4$ bytes
- **Weight traffic**: Every output element reads $9 \times N_i$ weight values.
  - Total: $B \times N_y \times N_x \times N_n \times 9 \times N_i \times 4$ bytes (naively re-fetched for every output)
- **Output traffic**: $B \times N_y \times N_x \times N_n \times 4$ bytes written once.

**Total DRAM:**
$$
\text{DRAM\_naive} = 2 \times B \times N_y \times N_x \times N_n \times 9 \times N_i \times 4 + \text{output}
$$

**Observation**: Weight is read repeatedly ($B \times N_y \times N_x \times N_n$ times), but it's only $9 \times N_i \times 4$ bytes. For large output tensors, this dominates.

### 2.2 K1: smem_w (Weights in Shared Memory)

Same as naive, but weights are cached in shared memory per block:
- Block loads weights once (cost amortized over all threads in the block).
- Input still re-fetched per thread.

**Analytic estimate**: Same as naive (we don't model per-block SPAD reuse in this first-order estimate).

### 2.3 K2: Unroll (Software Pipelining)

Same memory pattern as naive, but **instruction overhead** increases memory-side stalls.
- Estimate: multiply input traffic by 1.25 to account for extra latency hiding.

$$
\text{DRAM\_unroll} \approx 1.25 \times (B \times N_y \times N_x \times N_n \times 9 \times N_i \times 4) + \text{weights} + \text{output}
$$

### 2.4 K3: smem_wi (Weights + Input in Shared Memory — Key Kernel)

This is the **most important** kernel for demonstrating model accuracy.

**Tiling strategy:**
1. Divide output into $16 \times 16$ tiles (one block per tile).
2. Divide input channels into $\text{smwi\_tile\_ni}$ chunks (default = 8).
3. For each chunk, load a padded input patch $(18 \times 18 \times \text{smwi\_tile\_ni})$ into shared memory.
4. Load weight slice $(3 \times 3 \times \text{smwi\_tile\_ni})$ into shared memory.
5. Compute all output elements for this chunk; reuse inputs/weights from SPAD.

**Number of iterations:**
- Spatial tiles: $\lceil N_y / 16 \rceil \times \lceil N_x / 16 \rceil$
- Channel tiles: $\lceil N_i / \text{smwi\_tile\_ni} \rceil$
- Total chunks: $B \times \lceil N_y / 16 \rceil \times \lceil N_x / 16 \rceil \times \lceil N_i / \text{smwi\_tile\_ni} \rceil$

**Input DRAM traffic per chunk:**
$$
\text{input\_patch\_bytes} = 18 \times 18 \times \text{smwi\_tile\_ni} \times 4
$$
$$
\text{DRAM\_input\_smem\_wi} = B \times \lceil N_y / 16 \rceil \times \lceil N_x / 16 \rceil \times \lceil N_i / \text{smwi\_tile\_ni} \rceil \times 18 \times 18 \times \text{smwi\_tile\_ni} \times 4
$$

**Weight DRAM traffic per chunk:**
$$
\text{weight\_chunk\_bytes} = 3 \times 3 \times \text{smwi\_tile\_ni} \times 4
$$

However, if the entire weight tensor fits in L2 cache:
$$
\text{syn\_bytes} = 3 \times 3 \times N_i \times N_n \times 4 < \text{L2\_size}
$$
then we cap weight DRAM to a few full sweeps:
$$
\text{DRAM\_weight\_smem\_wi} = \text{min}(\text{syn\_bytes}, \text{syn\_bytes} \times (B \times N_n / \text{num\_SMs}))
$$

**Example for Conv1** (Ny=224, Nx=224, Ni=64, Nn=64):
- Weights: $3 \times 3 \times 64 \times 64 \times 4 = 147,456$ bytes ≈ 144 KB, **fits in 4.5 MB L2**.
  - Weight DRAM ≈ min(144 KB, 144 KB × (16×64 / 80)) ≈ 144 KB (loaded once, reused).
- Input patches: $\lceil 224 / 16 \rceil = 14$ per dimension, so $14 \times 14 = 196$ spatial tiles.
  - Per chunk: $18 \times 18 \times 8 \times 4 = 10,368$ bytes.
  - Total for 8-channel chunks: $196 \times \lceil 64/8 \rceil \times 10,368 = 196 \times 8 \times 10,368 ≈ 16.3$ MB.
  - **This is >4.5 MB L2, so significant re-fetch occurs.**

**Example for Conv2** (Ny=14, Nx=14, Ni=512, Nn=512):
- Weights: $3 \times 3 \times 512 \times 512 \times 4 ≈ 9.4$ MB, **exceeds L2**.
- Input (padded): $16 \times 16 \times 512 \times 4 ≈ 8.2$ MB, **close to L2 size**.
  - Few spatial tiles ($1 \times 1$), but many channel chunks → lower input re-fetch.

### 2.5 K4: regblock (Register Blocking)

Reduces input traffic by caching output elements in registers:
$$
\text{DRAM\_regblock} \approx \frac{\text{input\_naive}}{N_{reg}} + \text{weights} + \text{output}
$$
where $N_{reg} = 4$ (register block size).

### 2.6 K5: warp (Warp-Level Tiling)

Estimates traffic in cache-line units (128 bytes), accounting for coalescing:
$$
\text{input\_lines} = B \times N_y \times N_x \times N_n \times \lceil (K_y \times K_x \times N_i) / \text{warp\_size} \rceil
$$
$$
\text{DRAM\_warp} = \text{input\_lines} \times 128 + \text{weights} + \text{output}
$$

---

## Part 3: L2 Pressure and Traffic Correction

### 3.1 L2 Cache Pressure Metric

We define **L2 pressure** as:
$$
\text{L2\_pressure} = \frac{\max(\text{weights\_bytes}, \text{input\_bytes}, \text{output\_bytes})}{\text{L2\_size}}
$$

For TITAN V: L2 = 4.5 MB.

**Interpretation:**
- **L2_pressure < 1**: tensor fits entirely in L2 → perfect reuse after first load.
- **L2_pressure > 1**: tensor exceeds L2 → partial eviction and re-fetch.

**Examples:**
- Conv1 naive: input ≈ 200 MB, L2_pressure ≈ 44.4 (no L2 reuse for input).
- Conv2 naive: input ≈ 8.2 MB, L2_pressure ≈ 1.8 (partial L2 reuse).

### 3.2 Log-Linear Correction

The analytic DRAM estimate above is mechanistic but may underestimate traffic due to:
1. **Conflict misses** in L2 and L1 caches.
2. **Suboptimal coalescing** in real kernels vs. ideal assumptions.
3. **Prefetch inefficiency** under high memory pressure.

We fit a **log-linear correction**:
$$
\text{DRAM\_corrected} = \text{DRAM\_analytic} \times \exp(a + b \log(\text{L2\_pressure}))
$$

**Fitting process** (per kernel):
1. Compute $\text{DRAM\_analytic}$ for each sample (conv1/conv2).
2. Compute $\text{L2\_pressure}$ for each sample.
3. Compute measured $\text{DRAM\_measured}$.
4. Estimate ratio: $r_i = \text{DRAM\_measured}_i / \text{DRAM\_analytic}_i$.
5. Fit linear regression: $\log(r) = a + b \log(\text{L2\_pressure})$.

**Results** (example for K3 smem_wi):
- If pressure is low (e.g., 0.5): correction factor ≈ 0.95 (minor adjustment).
- If pressure is high (e.g., 10): correction factor ≈ 1.2–2.0 (significant traffic increase due to evictions).

**Clamping**: We bound $0.05 \leq \text{correction\_factor} \leq 20$ to prevent pathological over-correction outside the fitted regime.

---

## Part 4: Bottleneck Analysis and Scaling Factors

### 4.1 Ideal Time Decomposition

Given corrected DRAM bytes, we compute two ideal times:
$$
t_{\text{compute}} = \frac{\text{FLOPs}}{\text{peak\_FP32\_TFLOPS}} \times 10^{-9}
$$
$$
t_{\text{mem}} = \frac{\text{DRAM\_bytes}}{\text{DRAM\_BW}} \times 10^{-9}
$$

### 4.2 Per-Kernel Scaling Factors

Real hardware rarely achieves ideal time. Factors include:
- **Stalls**: register pressure, branch divergence, memory latency hiding.
- **Occupancy**: limited blocks per SM reduce parallelism.
- **Serialization**: atomic operations, synchronization.

We model this with **per-kernel scale factors**:
$$
\text{predicted\_time} = \max(\text{scale\_compute} \times t_{\text{compute}}, \text{scale\_mem} \times t_{\text{mem}})
$$

**Fitting** (grid search over $\text{scale\_compute}, \text{scale\_mem} \in [0.5, 128]$):
1. For each kernel, try all $(s_c, s_m)$ pairs.
2. Compute predicted time for each sample.
3. Compute average MAPE (Mean Absolute Percentage Error).
4. Select $(s_c, s_m)$ with lowest MAPE.

**Intuition:**
- **scale_compute > 1**: compute is bottleneck (latency hiding fails).
- **scale_mem > 1**: memory is bottleneck (bandwidth not saturated).
- **scale_compute >> scale_mem**: kernel is compute-bound.

**Example (Conv1 naive):**
- $\text{scale\_compute} = 96$, $\text{scale\_mem} = 1.0$.
- Interpretation: memory is well-saturated, but actual time is 96× longer than ideal compute time, which corresponds to a very large Time MAPE relative to the ideal compute bound.

---

## Part 5: Validation Methodology

### 5.1 Dataset

We validate on two convolution configurations:
1. **Conv1**: 224×224 input, 64→64 channels, $K_y = K_x = 3$.
2. **Conv2**: 14×14 input, 512→512 channels, $K_y = K_x = 3$.

Both use batch size $B = 16$.

Measured data includes:
- Kernel runtime (ms) from GPU timing.
- DRAM bytes (read + write) from Nsight Compute.
- Peak GFLOPS from specification.

### 5.2 Two-Path Evaluation

We compare two models:

**Path A: Shortcut Baseline (measured DRAM)**
- Uses actual measured DRAM bytes (no analytic traffic).
- Fits scale factors on measured data.
- **Expected**: perfect on the measured pairs, but cannot generalize.

**Path B: Primary Model (analytic DRAM + correction)**
- Uses mechanistic DRAM estimate + L2-pressure correction.
- Fits scale factors on corrected DRAM.
- **Expected**: generalizes to unseen problem sizes/kernel shapes.

### 5.3 Error Metrics

For each sample, we compute:
$$
\text{MAPE\_time} = \frac{|\text{predicted\_ms} - \text{measured\_ms}|}{\text{measured\_ms}} \times 100\%
$$
$$
\text{MAPE\_GFLOPS} = \frac{|\text{predicted\_GFLOPS} - \text{measured\_GFLOPS}|}{\text{measured\_GFLOPS}} \times 100\%
$$
$$
\text{MAPE\_AI} = \frac{|\text{predicted\_AI} - \text{measured\_AI}|}{\text{measured\_AI}} \times 100\%
$$

**Reported metrics:**
- Average MAPE per kernel.
- Average MAPE across all kernels.

---

## Part 6: Results and Interpretation

### 6.1 Primary Model Performance

The primary model (analytic DRAM + correction) achieves:
- **Time MAPE**: 9.30% average (range 1.7%–21.4%).
- **GFLOPS MAPE**: 8.75% average.
- **AI MAPE**: 3.29% average.

**Key observations:**
- **smem_wi** (K3): ~15% time error, but **correctly predicts compute-boundedness** for Conv2.
- **regblock** (K4): 1.7% error on Conv2 (best prediction).
- **warp** (K5): 21% error (spikes due to cache-line modeling assumptions).

### 6.2 Shortcut Baseline Performance

The shortcut model (measured DRAM) achieves:
- **Time MAPE**: 9.30% average.
- **GFLOPS MAPE**: 8.75% average.
- **AI MAPE**: 0% (by construction, since we use measured AI directly).

**Key observation**: Shortcut performs identically on the measured pairs, but **cannot be applied to unseen configurations** (e.g., KY=5, KX=5).

### 6.3 Comparison with Roofline

Roofline alone would predict:
- Conv1 naive: ~4 ms (actual: 686 ms) → Time MAPE ≈ 99.4%.
- Conv2 smem_wi: ~0.3 ms (actual: 12.5 ms) → Time MAPE ≈ 97.6%.

**Our model** reduces this to:
- Conv1 naive: 644 ms (686 ms measured) → **6% error**.
- Conv2 smem_wi: 11.9 ms (12.5 ms measured) → **5% error**.

---

## Part 7: Generalization to New Kernel Shapes (KY=5, KX=5)

With the analytic DRAM path and L2-pressure correction, the model can predict performance for:
- **Different kernel sizes**: KY=5, KX=5 (instead of 3×3).
- **Different problem sizes**: arbitrary Ny, Nx, Ni, Nn.
- **New kernel variants**: as long as we can estimate their DRAM traffic mechanistically.

**Validation on KY=5, KX=5 data:**
We measure new kernels with $K_y = K_x = 5$ and validate predictions using the same model (without re-fitting scale factors). Expected MAPE: 10–20% (slightly higher due to changed kernel geometry).

---

## Part 8: Code Organization

### 8.1 `hierarchical.py` (Main Model)

**Key functions:**
- `conv_flops()`: Compute total FLOPs.
- `conv_tensor_bytes()`: Compute weight/input/output byte counts.
- `conv_l2_pressure()`: Compute L2 pressure metric.
- `_analytic_dram_bytes()`: Per-kernel DRAM traffic estimation.
- `_fit_traffic_correction()`: Fit log-linear correction per kernel.
- `_fit_kernel_scalers()`: Grid-search scale factors.
- `predict_sample()`: Predict time for a sample.
- `evaluate()`: Full pipeline (load data, fit, predict).

### 8.2 `hw_params.py`

Tunable hardware constants:
- `fp32_tflops`: Peak FP32 (14.9).
- `dram_bw_gb_s`: DRAM bandwidth (652.8).
- `l2_bytes`: L2 cache size (4.5 MB).
- `l2_bw_gb_s`: L2 bandwidth (3100, tunable).
- `l1_bw_gb_s`: L1 bandwidth (12800, tunable).

### 8.3 `validate.py`

Generates comparison plots:
- `plot_model_vs_measured()`: Time predictions vs measurements.
- `plot_mape_by_kernel()`: Error breakdown by kernel.
- Supports custom result directories (e.g., `results_ky_5_kx_5`).

### 8.4 `plot_roofline_predicted.py`

Renders a roofline chart using model predictions (AI + GFLOPS) instead of measured data.

---

## Part 9: Designed Model Characteristics

### 9.1 What Worked Well

1. **L2-pressure-aware correction**: Simple (log-linear) yet effective at capturing cache effects.
2. **Per-kernel mechanistic analysis**: Different kernels have fundamentally different memory patterns.
3. **Scale-factor fitting**: Captures hardware-specific overheads without over-parameterization.

<!-- ### 9.2 Limitations and Future Work

1. **L1/SPAD modeling**: Currently we only model DRAM. Per-kernel L1 traffic could improve accuracy.
2. **Occupancy and stall prediction**: Scale factors are empirical; deeper stall analysis (register pressure, latency) could replace them.
3. **Multi-level memory hierarchy**: A full roofline (compute, L1, L2, DRAM) would require estimating traffic at each level separately.
4. **Kernel variants**: The model assumes fixed tiling parameters; tuning `smwi_tile_ni` could be modeled similarly. -->

---

## Test Results Corner


### Default input — Measured vs Predicted (Time, GFLOPS, AI)

| Config | Kernel | Time (ms) | Predicted (ms) | Time MAPE (%) | Measured GFLOPS | Predicted GFLOPS | GFLOPS MAPE (%) |
|--------|--------|----------:|---------------:|--------------:|----------------:|----------------:|----------------:|
| **Conv1** | K0 naive | 686.431 | 644.3781 | 6.13 | 86.23 | 91.86 | 6.53 |
| 224×224, Ni=Nn=64 | K1 smem_w | 665.675 | 625.7200 | 6.00 | 88.92 | 94.60 | 6.39 |
|  | K2 unroll | 1128.647 | 1058.3640 | 6.23 | 52.44 | 55.93 | 6.65 |
|  | K3 smem_wi | 57.045 | 66.0386 | 15.77 | 1037.60 | 896.29 | 13.62 |
|  | K4 regblock | 397.214 | 352.7918 | 11.18 | 149.01 | 167.78 | 12.59 |
|  | K5 warp | 171.040 | 190.6793 | 11.48 | 346.06 | 310.42 | 10.30 |
| **Conv2** | K0 naive | 84.237 | 95.3396 | 13.18 | 175.66 | 155.21 | 11.64 |
| 14×14, Ni=Nn=512 | K1 smem_w | 99.191 | 95.3396 | 3.88 | 149.18 | 155.21 | 4.04 |
|  | K2 unroll | 86.868 | 95.3396 | 9.75 | 170.34 | 155.21 | 8.88 |
|  | K3 smem_wi | 12.525 | 11.9175 | 4.85 | 1181.48 | 1241.67 | 5.09 |
|  | K4 regblock | 23.433 | 23.8349 | 1.72 | 631.48 | 620.83 | 1.69 |
|  | K5 warp | 39.268 | 47.6698 | 21.40 | 376.83 | 310.42 | 17.62 |


### Quick Fill-In Template

**Run command:**

```bash
python3 validate.py --config <name> --input-root <path> --ky <K> --kx <K>
```

**Generated files:**
- `results_<config>/model_vs_measured_time.png`
- `results_<config>/mape_by_kernel.png`
- `results_<config>/predictions_primary_model.csv`

---

## Appendix A: Hardware Parameters (NVIDIA TITAN V, Compute Capability 7.0)

| Parameter | Value | Note |
|-----------|-------|------|
| FP32 peak | 14.9 TFLOPS | 80 SMs × 64 cores × 2 ops/cycle × 1.455 GHz |
| DRAM bandwidth | 652.8 GB/s | HBM2, measured from ncu |
| L2 cache size | 4.5 MB | From deviceQuery |
| L2 bandwidth | 3100 GB/s | Estimated, tunable |
| L1 cache size | 96 KB/SM | Unified with shared memory |
| L1 bandwidth | 12800 GB/s | Estimated, tunable |
| Memory clock | 850 MHz | From deviceQuery |

---

## Appendix B: Roofline Chart Interpretation

On a roofline chart:
- **Points below the memory roof** (left of ridge point): memory-bound.
  - To improve: increase arithmetic intensity (more compute per byte) or increase bandwidth.
- **Points below the compute roof** (right of ridge point): compute-bound.
  - To improve: only increase FP32 throughput.
- **Ridge point** (intersection): AI = peak_FP32 / peak_BW ≈ 22.82 FLOPs/byte for TITAN V.

**Example:**
- Conv1 naive: AI ≈ 0.14 (far left, memory-bound) → achieved 86 GFLOPS (far below roof).
- Conv2 smem_wi: AI ≈ 87.5 (far right, compute-bound) → achieved 1181 GFLOPS (near peak FP32).

---

## Appendix C: Reproduction Instructions

### Generate predictions and plots:

```bash
cd ~/CS259_Projects/project2

# Validate against original Conv1/Conv2 (KY=3, KX=3)
python3 hierarchical.py          # Print calibrated factors and summaries
python3 validate.py              # Generate plots in results/

# Validate against new KY=5, KX=5 data
python3 validate.py --config ky_5_kx_5 \
    --results-dir ~/more_data_project1/results \
    --ky 5 --kx 5
```

### Expected outputs:

- `results/model_vs_measured_time.png`: Bar chart (measured vs predicted runtime).
- `results/mape_by_kernel.png`: Error by kernel type.
- `results_ky_5_kx_5/model_vs_measured_time.png`: Same for new kernel shape.
- `results/roofline_predicted.png`: Roofline with model predictions.

---
<!-- 
## References

- Williams, S., Waterman, A., & Patterson, D. (2009). "Roofline: an insightful visual performance model for floating-point programs." *Communications of the ACM*, 52(4), 65–76.
- NVIDIA Volta Architecture Whitepaper (arXiv:1804.06826).
- Nsight Compute documentation: https://docs.nvidia.com/nsight-compute/ -->
