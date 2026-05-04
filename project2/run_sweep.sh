#!/usr/bin/env bash
# =============================================================================
# run_sweep.sh  –  CS259 Mini-proj 2: sweep problem sizes and tile sizes.
#
# Produces CSV files in results/measured/ for use in validate.py.
#
# Usage:
#   ./experiments/run_sweep.sh
#
# Requires: ./conv binary compiled from Mini-proj 1 with parameterised SMWI_TILE_NI.
# See note below about recompiling for tile sweeps.
# =============================================================================

set -euo pipefail

BINARY=../conv           # adjust path as needed
RESULTS=../results/measured
mkdir -p "$RESULTS"

# ── Step 1: Problem size sweep (spatial size, K3 smem_wi) ───────────────────
# Run Conv1-like configs at different spatial sizes (Ny=Nx varies).
# These require modifying main() to accept B, Ny, Nx, Ni, Nn as args,
# OR create a small sweep binary.
echo "=== Problem size sweep (K3 smem_wi) ==="
echo "config,Ny,Nx,Ni,Nn,B,kernel,time_ms" > "$RESULTS/problem_size_sweep.csv"

# Placeholder: run your binary and grep timing lines.
# Replace the loop body with actual binary calls.
for SIZE in 16 32 56 112 224; do
    echo "  → Ny=Nx=$SIZE"
    # Example call — you may need to modify conv.cu to accept size args:
    # TIME=$("$BINARY" $SIZE | grep smem_wi | awk '{print $2}')
    # echo "conv1_like,$SIZE,$SIZE,64,64,16,smem_wi,$TIME" >> "$RESULTS/problem_size_sweep.csv"
    echo "# TODO: run binary for Ny=Nx=$SIZE and record time" >&2
done

# ── Step 2: Tile size sweep (smwi_tile_ni) ──────────────────────────────────
# This requires recompiling conv.cu with different SMWI_TILE_NI values.
# Easiest: add a build target in Makefile for each tile size.
#
# To parametrise SMWI_TILE_NI at compile time:
#   nvcc ... -DSMWI_TILE_NI=4 -o conv_tni4 conv.cu
echo ""
echo "=== Tile size sweep (smwi_tile_ni) ==="
echo "config,smwi_tile_ni,time_ms" > "$RESULTS/tile_size_sweep.csv"

for TNI in 1 2 4 8 16 32 64; do
    echo "  → smwi_tile_ni=$TNI"
    # Recompile with different tile size:
    # nvcc -O3 -arch=sm_70 -DSMWI_TILE_NI=$TNI -o conv_tni conv.cu && \
    #   TIME=$(./conv_tni 1 3 | grep smem_wi | awk '{print $2}')
    # echo "conv1,$TNI,$TIME" >> "$RESULTS/tile_size_sweep.csv"
    echo "# TODO: compile with SMWI_TILE_NI=$TNI and record time" >&2
done

# ── Step 3: ncu metrics for intermediate validation ──────────────────────────
# Compare predicted DRAM bytes against measured dram__bytes_read.sum
echo ""
echo "=== ncu DRAM traffic validation ==="
METRICS="dram__bytes_read.sum,dram__bytes_write.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed"
for layer in 1 2; do
    for kid in 0 3; do   # naive and smem_wi
        label="conv${layer}_$([ $kid -eq 0 ] && echo naive || echo smem_wi)"
        echo "  → $label"
        # ncu --kernel-name "$([ $kid -eq 0 ] && echo conv_naive_kernel || echo conv_smem_wi_kernel)" \
        #     --launch-count 1 --metrics "$METRICS" \
        #     "$BINARY" "$layer" "$kid" > "$RESULTS/${label}_dram.txt" 2>&1
    done
done

echo ""
echo "Done. Fill in CSVs, then run: python3 analysis/validate.py"
