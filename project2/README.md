# CS259 Mini-Project 2 — Performance Modeling Template

## Kernel: Conv (K3 smem_wi focus, all 6 kernels modeled)

---

## Project Structure

```
project2/
├── model/
│   ├── hw_params.py         # TITAN V hardware constants (tunable)
│   ├── roofline.py          # Baseline roofline (Mini-proj 1 extension)
│   └── hierarchical.py      # Main mechanistic model (K0–K5)
├── analysis/
│   └── validate.py          # Validation plots (all 4 required graphs)
├── experiments/
│   └── run_sweep.sh         # GPU sweep script (TODO: fill in binary calls)
└── results/
    ├── measured/            # CSV from GPU runs
    └── predicted/           # Model outputs
```

---

## Model Design (K3 smem_wi — Primary Kernel)

The model performs a **bottleneck analysis** at three memory hierarchy levels:

```
predicted_time = max(compute_time, dram_time, l2_time, spad_time)

where:
  compute_time = total_FLOPs / peak_FP32
  dram_time    = actual_DRAM_bytes / DRAM_BW
  l2_time      = l2_bytes_traffic / L2_BW
  spad_time    = spad_bytes_traffic / L1_BW
```

### DRAM Traffic for K3 smem_wi

The key insight is tiling: for each `smwi_tile_ni`-wide ni chunk:
1. Load `18×18×smwi_tile_ni` input patch into shared memory
2. Load `KY×KX×smwi_tile_ni` weight slice into shared memory  
3. Compute 16×16 output elements from SPAD — no DRAM access

**Input DRAM traffic:**
```
= B × ceil(Ny/16) × ceil(Nx/16) × ceil(Ni/smwi_tile_ni) × 18×18×smwi_tile_ni × 4
```
Reduced further by L2 hit rate when input fits in L2 (Conv2's 8 MB just exceeds 4.5 MB L2).

**Weight DRAM traffic:**
- Conv1: 144 KB weight fits in L2 → loaded once, reused across spatial tiles
- Conv2: 9.4 MB weight > L2 → partial DRAM re-fetch

---

## TODO Checklist (what you need to do)

### Phase 1 — Validate against existing Mini-proj 1 data ✅ (already have)
- [x] Conv1 naive: 685.6 ms measured
- [x] Conv1 smem_wi: 57.2 ms measured  
- [x] Conv2 smem_wi: 12.5 ms measured

### Phase 2 — Tune model parameters
Run `python3 model/hierarchical.py` and compare MAPE to ncu measurements.
Key tuning knobs in `hw_params.py`:
- `l2_bw_gb_s` — most uncertain, start at 3100 GB/s
- `l1_bw_gb_s` — adjust until spad_time matches for smem_wi

### Phase 3 — GPU sweep (modify conv.cu)
1. **Problem size sweep**: modify `main()` to accept `Ny, Nx` as args
   ```bash
   ./conv_sweep [layer] [kernel_id] [Ny] [Nx]
   ```
2. **Tile size sweep**: make `SMWI_TILE_NI` a compile-time `-D` flag
   ```makefile
   conv_tni%: conv.cu
       nvcc -O3 -arch=sm_70 -DSMWI_TILE_NI=$* -o $@ $<
   ```
3. Fill in `measured_by_tile_ni` and `measured_times` in `analysis/validate.py`

### Phase 4 — Generate all plots
```bash
cd analysis && python3 validate.py
```
This generates:
- `results/error_vs_problem_size.png`
- `results/error_vs_tile_size.png`
- `results/model_vs_roofline.png`
- `results/sensitivity_analysis.png`

---

## Key Observations to Discuss in Report

### Why smem_wi is so good (and why the model must capture this)
- Conv1 naive: actual AI = 0.141 (roofline predicts ~86 GFLOPS) ✓
- Conv1 smem_wi: actual AI = 2.059 — input reuse 93% reduces DRAM traffic
- Conv2 smem_wi: actual AI = 87.49 — crosses ridge point (22.82) → compute-bound
  - The model MUST predict compute-bound here, or it's wrong about the regime

### The L2 boundary effect (important for model accuracy)
- Conv1 input (199 MB) >> L2 (4.5 MB) → no L2 reuse for input → pure DRAM
- Conv2 input (8 MB) ≈ L2 (4.5 MB) → partial L2 reuse for naive kernel
- This explains why naive is faster for Conv2 than Conv1 (relative to roofline)

### Tile size effect on smem_wi
- Larger smwi_tile_ni → fewer ni-chunks → fewer DRAM loads → less DRAM traffic
- But larger smwi_tile_ni → larger smem → fewer blocks/SM → less occupancy
- There's an optimal smwi_tile_ni: the model should predict this tradeoff

### Architecture Sensitivity (for report section 3)
- Conv2 smem_wi is compute-bound → only FP32 TFLOPS improvement helps
- Conv1 smem_wi is still memory-bound → DRAM BW improvement helps
- Future GPU insight: more on-chip memory (SPAD) would help Conv1 by enabling
  larger tiles → more input reuse → higher AI

---

## Roofline Comparison

Your model should beat roofline on MAPE for smem_wi because:
- Roofline uses theoretical minimum DRAM (each byte once) → always too optimistic
- Your model accounts for actual DRAM traffic from tiling analysis
- Example: Conv1 roofline predicts ~4 ms; actual is 57 ms → 14× error
           Your hierarchical model should get within ~2× by modeling input re-fetch

---

## Hardware Parameters to Tune

| Parameter       | Default        | Note                                          |
|-----------------|----------------|-----------------------------------------------|
| `dram_bw_gb_s`  | 652.8          | Cross-checked from ncu — keep as-is           |
| `l2_bw_gb_s`    | 3100           | Estimate — tune against smem_wi data          |
| `l1_bw_gb_s`    | 12800          | Estimate — tune against smem_wi SPAD-limited  |
| `fp32_tflops`   | 14.9           | From spec — keep as-is                        |
| `num_sms`       | 80             | From spec — keep as-is                        |
