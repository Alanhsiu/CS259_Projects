#!/bin/bash
source ~/myenv/bin/activate

set -e

BINARY=./attention
RESULTS=./results
NCU=ncu

mkdir -p "$RESULTS"

METRICS="dram__bytes_read.sum,\
dram__bytes_write.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

run_ncu() {
    local label=$1
    local kernel=$2
    local mode=$3
    local size=$4
    echo "Profiling: $label  (kernel=$kernel  mode=$mode  size=$size)"

    "$NCU" \
        --kernel-name "$kernel" \
        --launch-skip 3 \
        --launch-count 1 \
        --set full \
        -o "$RESULTS/$label" \
        "$BINARY" "$mode" "$size" 2>/dev/null

    "$NCU" \
        --kernel-name "$kernel" \
        --launch-skip 3 \
        --launch-count 1 \
        --metrics "$METRICS" \
        "$BINARY" "$mode" "$size" \
        > "$RESULTS/${label}_metrics.txt" 2>&1

    "$NCU" \
        --kernel-name "$kernel" \
        --launch-skip 3 \
        --launch-count 1 \
        --set full \
        --print-summary per-kernel \
        "$BINARY" "$mode" "$size" \
        > "$RESULTS/${label}_full_log.txt" 2>&1

    echo "  -> saved $RESULTS/$label.*"
}

echo "========================================"
echo "  Building..."
echo "========================================"
make -j

echo ""
echo "========================================"
echo "  Profiling PREFILL S=4096"
echo "========================================"
run_ncu "prefill4096_naive"    "prefill_naive_kernel"    prefill 4096
run_ncu "prefill4096_flash"    "prefill_flash_kernel"    prefill 4096
run_ncu "prefill4096_flash_v2" "prefill_flash_v2_kernel" prefill 4096

echo ""
echo "========================================"
echo "  Profiling PREFILL S=65536"
echo "========================================"
run_ncu "prefill65536_flash"    "prefill_flash_kernel"    prefill 65536
run_ncu "prefill65536_flash_v2" "prefill_flash_v2_kernel" prefill 65536

echo ""
echo "========================================"
echo "  Profiling DECODE C=4096"
echo "========================================"
run_ncu "decode4096_naive"       "decode_naive_kernel"        decode 4096
run_ncu "decode4096_flash_p1"    "decode_flash_p1_kernel"     decode 4096
run_ncu "decode4096_flash_v2_p1" "decode_flash_v2_p1_kernel"  decode 4096

echo ""
echo "========================================"
echo "  Profiling DECODE C=65536"
echo "========================================"
run_ncu "decode65536_naive"       "decode_naive_kernel"        decode 65536
run_ncu "decode65536_flash_p1"    "decode_flash_p1_kernel"     decode 65536
run_ncu "decode65536_flash_v2_p1" "decode_flash_v2_p1_kernel"  decode 65536

echo ""
echo "========================================"
echo "  Collecting benchmark summary"
echo "========================================"
"$BINARY" all > "$RESULTS/summary.txt" 2>&1
cat "$RESULTS/summary.txt"

echo ""
echo "All done. Results in $RESULTS/"
