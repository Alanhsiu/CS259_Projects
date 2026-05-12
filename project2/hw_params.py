"""
hw_params.py  –  NVIDIA TITAN V (CC 7.0) hardware constants.

Values grounded in:
  - Volta Architecture Whitepaper (arXiv 1804.06826)
  - ncu measurements from Mini-proj 1 (e.g., DRAM BW cross-checked at 94% utilisation)
  - Measured: Conv1 naive reads 420 GB in 686 ms → 613 GB/s = 93.9% × 652.8 GB/s ✓
"""

HW = {
  # ── Device Identity (from deviceQuery) ──────────────────────────────────
  "gpu_name":             "NVIDIA TITAN V",
  "cuda_cc":              "7.0",
  "cuda_driver_version":  "12.4",
  "cuda_runtime_version": "12.4",

    # ── Compute ──────────────────────────────────────────────────────────────
    "num_sms":              80,
    "cuda_cores_per_sm":    64,       # 4 warp schedulers × 16 FP32 pipes
    "clock_ghz":            1.455,    # boost clock
    "fp32_tflops":          14.9,     # peak FP32 (80 SM × 64 × 2 × 1.455 GHz)
    "fp32_flops_per_sm":    14.9e12 / 80,

    # ── Memory Hierarchy ─────────────────────────────────────────────────────
    "global_mem_bytes":     12_635_406_336,  # exact from deviceQuery
    "memory_clock_mhz":     850,
    "memory_bus_width_bits":3072,
    # DRAM: confirmed from ncu (see Mini-proj 1 cross-check)
    "dram_bw_gb_s":         652.8,    # HBM2 peak bandwidth
    # L2: estimated from Volta whitepaper (slice bandwidth × 8 slices)
    "l2_bw_gb_s":           3_100.0,  # tunable — start here
    # L1/shared memory: measured in literature as ~6× L2 per SM
    "l1_bw_gb_s":           12_800.0, # tunable

    # Latencies (cycles, Volta)
    "dram_latency_cyc":     800,
    "l2_latency_cyc":       200,
    "l1_latency_cyc":       28,

    # ── Capacities ───────────────────────────────────────────────────────────
    "l2_bytes":             4_718_592,  # exact from deviceQuery (4.5 MiB)
    "l1_spad_bytes_per_sm": 96 * 1024,  # 96 KB unified L1+shared per SM (configurable)
    "max_shared_per_block": 48 * 1024,  # default max per block (can raise to 96 KB)
    "reg_file_per_sm":      256 * 1024, # 256 KB register file per SM

    # ── Thread limits ─────────────────────────────────────────────────────────
    "max_threads_per_sm":   2048,
    "max_threads_per_block":1024,
    "max_blocks_per_sm":    32,
    "warp_size":            32,
}

# ── Kernel-specific tile parameters (from conv.cu constants) ─────────────────
KERNEL_PARAMS = {
    "K0_naive":   {"tile_x": 16, "tile_y": 16, "smem_bytes": 0},
    "K1_smem_w":  {"tile_x": 16, "tile_y": 16,  # smem = KY*KX*Ni*4
                   "smem_bytes": lambda Ni: 3*3*Ni*4},
    "K3_smem_wi": {"tile_x": 16, "tile_y": 16,
                   "smwi_tile_ni": 8,
                   # smem = (KY*KX*8 + 18*18*8)*4 = 10,656 bytes
                   "smem_bytes": lambda: (3*3*8 + 18*18*8)*4},
    "K4_regblock":{"tile_x": 16, "tile_y": 16, "reg_n": 4},
    "K5_warp":    {"warp_tile_x": 8, "warp_size": 32},
}
