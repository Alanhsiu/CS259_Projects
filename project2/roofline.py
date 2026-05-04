"""
roofline.py  –  Baseline roofline model for CS259 Mini-proj 2.

This is the SIMPLEST valid model — your starting point.
It assumes:
  - Perfect compute (peak FP32 TFLOPS)
  - Perfect DRAM bandwidth
  - Every tensor byte read/written exactly once (theoretical minimum traffic)
  - No hierarchy — just DRAM vs compute

Your hierarchical model in hierarchical.py should beat this significantly.
"""

from hw_params import HW


def tensor_bytes(B, Ny, Nx, Ni, Nn, dtype_bytes=4):
    """Minimum DRAM traffic: read each tensor once, write output once."""
    KY, KX = 3, 3
    NYPAD = Ny + KY - 1
    NXPAD = Nx + KX - 1
    syn_bytes = KY * KX * Nn * Ni * dtype_bytes
    inp_bytes = B * NYPAD * NXPAD * Ni * dtype_bytes
    out_bytes = B * Ny * Nx * Nn * dtype_bytes
    return syn_bytes, inp_bytes, out_bytes


def total_flops(B, Ny, Nx, Ni, Nn):
    """Total FLOPs: 2 × FMAs (matches ncu smsp__sass_thread_inst_executed_op_ffma_pred_on × 2)."""
    KY, KX = 3, 3
    return 2 * B * Ny * Nx * Nn * KY * KX * Ni


def roofline(B, Ny, Nx, Ni, Nn, dtype_bytes=4):
    """
    Classic roofline: predict time = max(compute_time, memory_time).
    Uses theoretical minimum DRAM traffic (each byte once).
    """
    flops = total_flops(B, Ny, Nx, Ni, Nn)
    syn_b, inp_b, out_b = tensor_bytes(B, Ny, Nx, Ni, Nn, dtype_bytes)
    total_bytes = syn_b + inp_b + out_b

    peak_flops = HW["fp32_tflops"] * 1e12
    peak_bw    = HW["dram_bw_gb_s"] * 1e9
    ridge      = peak_flops / peak_bw  # ~22.82 FLOPs/byte

    compute_time_s = flops / peak_flops
    memory_time_s  = total_bytes / peak_bw
    predicted_time_s = max(compute_time_s, memory_time_s)

    theoretical_ai = flops / total_bytes

    return {
        "predicted_time_ms": predicted_time_s * 1e3,
        "tflops":            flops / predicted_time_s / 1e12,
        "theoretical_ai":    theoretical_ai,
        "bottleneck":        "compute" if compute_time_s >= memory_time_s else "memory",
        "ridge_point":       ridge,
        "total_flops":       flops,
        "total_bytes_dram":  total_bytes,
    }


# ── Quick self-test against Mini-proj 1 known values ─────────────────────────
if __name__ == "__main__":
    configs = [
        ("Conv1", 16, 224, 224,  64,  64),
        ("Conv2", 16,  14,  14, 512, 512),
    ]

    print(f"\n{'Config':<8} {'Theory AI':>12} {'Roof Time(ms)':>14} {'Bottleneck'}")
    print("-" * 52)
    for name, B, Ny, Nx, Ni, Nn in configs:
        r = roofline(B, Ny, Nx, Ni, Nn)
        print(f"{name:<8} {r['theoretical_ai']:>12.1f} {r['predicted_time_ms']:>14.2f}"
              f"  {r['bottleneck']}")

    # Expected: Conv1 theory AI ≈ 142.7, Conv2 ≈ 610.2 (matches Mini-proj 1 report)
