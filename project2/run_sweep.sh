#!/usr/bin/env bash
# =============================================================================
# run_sweep.sh  –  CS259 Project 2: sweep problem sizes and tile sizes.
#
# Produces CSV files in results/measured/ for use in validate.py.
#
# Usage:
#   ./run_sweep.sh
#
# Requires: 
#   - ./conv binary compiled from conv.cu
#   - ./conv_tni_* binaries for tile size sweeps
# =============================================================================

set -euo pipefail

RESULTS=results/measured
mkdir -p "$RESULTS"

# ── Step 1: Problem size sweep (spatial size, K3 smem_wi) ───────────────────
echo "=== Problem size sweep (K3 smem_wi) ==="
echo "config,Ny,Nx,Ni,Nn,B,kernel,time_ms" > "$RESULTS/problem_size_sweep.csv"

# Build the binary if needed
if [ ! -f conv ]; then
    echo "Building conv binary..."
    make
fi

for SIZE in 16 32 56 112 224; do
    echo "  → Ny=Nx=$SIZE"
    # Run with layer=1 (Conv1), kernel=3 (smem_wi)
    # Extract the timing (last number in the kernel output line)
    TIME_LINE=$("./conv" 1 3 "$SIZE" "$SIZE" 2>/dev/null | grep "smem_wi" || echo "")
    if [ -n "$TIME_LINE" ]; then
        # Extract timing in ms (second field)
        TIME_MS=$(echo "$TIME_LINE" | awk '{print $2}')
        echo "conv1,$SIZE,$SIZE,64,64,16,smem_wi,$TIME_MS" >> "$RESULTS/problem_size_sweep.csv"
        echo "    Time: ${TIME_MS}ms"
    else
        echo "    ERROR: no timing extracted"
    fi
done

# ── Step 2: Tile size sweep (smwi_tile_ni) ──────────────────────────────────
echo ""
echo "=== Tile size sweep (smwi_tile_ni) ==="
echo "config,smwi_tile_ni,time_ms" > "$RESULTS/tile_size_sweep.csv"

# Build tile_ni binaries if needed
if [ ! -f conv_tni_2 ]; then
    echo "Building tile_ni binaries..."
    make tile_ni
fi

for TNI in 2 4 8 16 32; do
    echo "  → smwi_tile_ni=$TNI"
    BINARY="conv_tni_${TNI}"
    # Run with layer=1 (Conv1), kernel=3 (smem_wi), default sizes (224x224)
    TIME_LINE=$("./${BINARY}" 1 3 2>/dev/null | grep "smem_wi" || echo "")
    if [ -n "$TIME_LINE" ]; then
        TIME_MS=$(echo "$TIME_LINE" | awk '{print $2}')
        echo "conv1,$TNI,$TIME_MS" >> "$RESULTS/tile_size_sweep.csv"
        echo "    Time: ${TIME_MS}ms"
    else
        echo "    ERROR: no timing extracted"
    fi
done

echo ""
echo "Done. Results in $RESULTS/"
echo "Now run: python3 validate.py"

