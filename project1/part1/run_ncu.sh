#!/usr/bin/env bash
# =============================================================================
# run_ncu.sh  –  Profile all kernel variants (K0–K5) for Conv1 and Conv2.
#
# Output in results/:
#   summary.txt                  – timing table (all kernels, no profiler)
#   <label>_metrics.txt          – raw ncu metric values
#   <label>.ncu-rep              – full ncu report (open with ncu-ui)
#   roofline.png                 – Roofline chart
# =============================================================================

set -euo pipefail

BINARY=./conv
RESULTS=results
mkdir -p "$RESULTS"

# ncu reports Gbyte in SI units (1 Gbyte = 1e9 bytes); confirmed by cross-check
# with gpu__dram_throughput %.
METRICS="\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum"

# Columns: layer  kernel_id  kernel_func_name       label
declare -a CONFIGS=(
    "1  0  conv_naive_kernel     conv1_naive"
    "1  1  conv_smem_kernel      conv1_smem_w"
    "1  2  conv_unroll_kernel    conv1_unroll"
    "1  3  conv_smem_wi_kernel   conv1_smem_wi"
    "1  4  conv_regblock_kernel  conv1_regblock"
    "1  5  conv_warp_kernel      conv1_warp"
    "2  0  conv_naive_kernel     conv2_naive"
    "2  1  conv_smem_kernel      conv2_smem_w"
    "2  2  conv_unroll_kernel    conv2_unroll"
    "2  3  conv_smem_wi_kernel   conv2_smem_wi"
    "2  4  conv_regblock_kernel  conv2_regblock"
    "2  5  conv_warp_kernel      conv2_warp"
)

# ── Step 1: Timing (no profiler overhead) ────────────────────────────────────
echo "======================================================================"
echo "  Step 1: Timing summary (no profiler overhead)"
echo "======================================================================"
"$BINARY" | tee "$RESULTS/summary.txt"
echo ""

# ── Step 2: ncu specific metrics ─────────────────────────────────────────────
echo "======================================================================"
echo "  Step 2: ncu metric collection"
echo "======================================================================"

for cfg in "${CONFIGS[@]}"; do
    read -r layer kid kfunc label <<< "$cfg"
    out_txt="$RESULTS/${label}_metrics.txt"
    echo ""
    echo "  → $label  (layer=$layer, kernel_id=$kid)"
    TMPDIR="${TMPDIR:-$HOME/tmp}"; mkdir -p "$TMPDIR"
    TMP="$TMPDIR" ncu \
        --kernel-name "$kfunc" \
        --launch-count 1 \
        --metrics "$METRICS" \
        "$BINARY" "$layer" "$kid" \
        > "$out_txt" 2>&1
    echo "    Saved → $out_txt"
done

# ── Step 3: ncu full reports ──────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  Step 3: ncu full reports (--set full)"
echo "======================================================================"

for cfg in "${CONFIGS[@]}"; do
    read -r layer kid kfunc label <<< "$cfg"
    echo ""
    echo "  → $label"
    TMPDIR="${TMPDIR:-$HOME/tmp}"
    TMP="$TMPDIR" ncu \
        --kernel-name "$kfunc" \
        --launch-count 1 \
        --set full \
        -o "$RESULTS/${label}" \
        "$BINARY" "$layer" "$kid" \
        > "$RESULTS/${label}_full_log.txt" 2>&1
    echo "    Saved → $RESULTS/${label}.ncu-rep"
done

# ── Step 4: Roofline PNG ──────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  Step 4: Generating roofline.png"
echo "======================================================================"
python3 plot_roofline.py "$RESULTS"

echo ""
echo "======================================================================"
echo "  Done.  Files in: $RESULTS/"
echo "======================================================================"
