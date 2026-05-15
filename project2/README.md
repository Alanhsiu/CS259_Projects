# CS259 Project 2 — Hierarchical Performance Model for Convolution Kernels

This project models the Conv kernels from project 1 with a calibrated hierarchy that is more detailed than a simple roofline baseline.

## Repository Layout

```
project2/
├── conv                 # CUDA benchmark binary
├── conv.cu              # Kernel implementation
├── conv_ref.h           # Reference convolution helpers
├── hierarchical.py      # Main hierarchical model and calibration script
├── hw_params.py         # Hardware constants for TITAN V
├── validate.py          # Plot generation and comparison helpers
├── Makefile             # Build targets for conv and tile sweeps
├── run_sweep.sh         # Convenience script for GPU sweeps
├── report.md            # Final writeup
├── results/             # Generated predictions and plots
└── README.md
```

## Model Hierarchy

The current model has two prediction paths:

1. **Primary model**: analytically estimates DRAM traffic, applies a small shape-aware correction, then predicts runtime with a three-level bottleneck hierarchy.
2. **Shortcut baseline**: uses measured DRAM bytes directly so the calibrated kernel scalers can be compared against a simpler traffic source.

The runtime prediction is dominated by the slowest of these levels:

```text
predicted_time = max(compute_time, l2_time, dram_time)
```

`compute_time` is derived from peak FP32 throughput, `l2_time` models reuse that is served from L2, and `dram_time` models off-chip traffic. The script also tracks L1/shared-memory pressure for diagnostics and tile sweeps, but the main bottleneck decision is compute vs L2 vs DRAM.

### Kernel Order

The model covers all six convolution kernels used in project 1:

- `naive`
- `smem_w`
- `unroll`
- `smem_wi`
- `regblock`
- `warp`

### K3 `smem_wi` Focus

The tile sweep analysis is built around `smem_wi`, since it exposes the clearest hierarchy effects:

- smaller `smwi_tile_ni` improves coalescing less efficiently but uses less shared memory
- larger `smwi_tile_ni` reduces input re-fetches but lowers occupancy
- the model uses this tradeoff to explain the U-shaped performance curve

## Scripts

- `python3 hierarchical.py` fits the model against the project 1 measurements and writes `results/predictions_primary_model.csv` and `results/predictions_shortcut_model.csv`.
- `python3 validate.py` generates the comparison plots in `results/`.
- `make` builds the default `conv` binary.
- `make tile_ni` builds the `conv_tni_2`, `conv_tni_4`, `conv_tni_8`, `conv_tni_16`, and `conv_tni_32` binaries for tile sweeps.

## Generated Outputs

The main outputs produced by the model and validation flow are:

- `results/predictions_primary_model.csv`
- `results/predictions_shortcut_model.csv`
- `results/model_vs_measured_time.png`
- `results/model_vs_roofline.png`
- `results/error_vs_tile_size.png`
- `results/error_vs_problem_size.png`

## Notes

- The calibration data comes from the project 1 part 1 summary and Nsight Compute DRAM metrics.
- The primary model is meant to generalize to new convolution shapes through the analytic DRAM path.
- `report.md` contains the detailed derivation and the final discussion of the hierarchical model.
