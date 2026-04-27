#!/usr/bin/env python3
"""Roofline plot — Attention kernels on NVIDIA TITAN V."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os

PEAK_FP32 = 14_900.0   # GFLOPS
PEAK_BW   = 652.8      # GB/s
RIDGE     = PEAK_FP32 / PEAK_BW  # ~21.1

# ─── Measured data ───────────────────────────────────────────────────────────
# (label, AI [FLOPs/byte], GFLOPS, color, marker, label_offset_pts)
# AI = (ffma × 2) / (dram_read_bytes + dram_write_bytes)

RED    = "#D62728"
BLUE   = "#1F77B4"
GREEN  = "#2CA02C"
PURPLE = "#7B2D8B"
ORANGE = "#E67E22"
TEAL   = "#17A589"

# Small horizontal jitter to separate points with nearly identical AI
EPS = 0.97  # multiply AI by this to nudge left

points = [
    # label,                        AI,                                    GF,    color,  mk,  (dx,dy,ha,va)
    # ── PREFILL S=4096 ──────────────────────────────────────────────────────
    # P-naive: ffma=1149M dram=81.64MB
    ("P-naive\nS=4k",   2*1_149_519_872/81.64e6,       292, RED,    "o", ( 6, 4,"left","bottom")),
    # P-flash: ffma=5371M dram=4.34MB
    ("P-flash\nS=4k",   2*5_371_330_560/4.34e6,        395, BLUE,   "o", ( 6, 4,"left","bottom")),
    # P-flash_v2: ffma=5371M dram=4.35MB (nudge left so labels don't stack)
    ("P-flash_v2\nS=4k",2*5_371_330_560/4.35e6*EPS,    255, GREEN,  "o", (-6, 4,"right","bottom")),

    # ── PREFILL S=65536 ─────────────────────────────────────────────────────
    # P-flash: ffma=1374B dram=742.74GB
    ("P-flash\nS=64k",  2*1_374_431_477_760/742.74e9,  391, BLUE,   "s", (-6, 4,"right","bottom")),
    # P-flash_v2: ffma=1374B dram=11.578GB
    ("P-flash_v2\nS=64k",2*1_374_431_477_760/11.578e9, 258, GREEN,  "s", ( 6,-8,"left","top")),

    # ── DECODE C=4096 ───────────────────────────────────────────────────────
    # D-naive: ffma=561K dram=2.10MB
    ("D-naive\nC=4k",   2*561_152/2.10e6,              2.52, ORANGE, "o", ( 6, 4,"left","bottom")),
    # D-flash: ffma=2621K dram=2.101MB  (nudge left)
    ("D-flash\nC=4k",   2*2_621_440/2.101e6*EPS,       9.66, PURPLE, "o", (-6, 4,"right","bottom")),
    # D-flash_v2: ffma=2621K dram=2.111MB
    ("D-flash_v2\nC=4k",2*2_621_440/2.111e6,           9.85, TEAL,   "o", ( 6,-8,"left","top")),

    # ── DECODE C=65536 ──────────────────────────────────────────────────────
    # D-naive: ffma=8978K dram=35.86MB
    ("D-naive\nC=64k",  2*8_978_432/35.86e6,           1.93, ORANGE, "s", ( 6,-10,"left","top")),
    # D-flash: ffma=41943K dram=35.00MB (nudge left)
    ("D-flash\nC=64k",  2*41_943_040/35.00e6*EPS,      90.0, PURPLE, "s", (-6,  4,"right","bottom")),
    # D-flash_v2: ffma=41943K dram=35.00MB
    ("D-flash_v2\nC=64k",2*41_943_040/35.00e6,         91.5, TEAL,   "s", ( 6, -8,"left","top")),
]

# ─── Figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))

ai_x = np.logspace(-2, 5, 1000)
ax.loglog(ai_x, np.minimum(PEAK_BW * ai_x, PEAK_FP32), "k-", lw=2.5, zorder=3)
ax.axvline(RIDGE, color="gray", ls="--", lw=1.2, zorder=3)

# annotations: peak labels on roofline
ax.text(500, PEAK_FP32*1.15, "Peak: 14.9 TFLOPS", fontsize=9, ha="left")
ax.text(0.12, 14, f"Slope = {PEAK_BW:.0f} GB/s", fontsize=9, rotation=38, color="#333")
ax.text(RIDGE*1.05, 0.75, f"Ridge\n≈{RIDGE:.1f}", fontsize=8, color="gray", va="bottom")

# ─── Plot points + labels ────────────────────────────────────────────────────
for label, ai, gf, color, mk, (dx, dy, ha, va) in points:
    ax.scatter(ai, gf, color=color, marker=mk, s=140,
               edgecolors="k", linewidths=0.8, zorder=5)
    ax.annotate(
        label, (ai, gf),
        textcoords="offset points", xytext=(dx, dy),
        fontsize=8, ha=ha, va=va,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, ec="none"),
        zorder=6,
    )

# ─── Legend ──────────────────────────────────────────────────────────────────
legend_handles = [
    Patch(color=RED,    label="Prefill naive"),
    Patch(color=BLUE,   label="Prefill flash"),
    Patch(color=GREEN,  label="Prefill flash_v2"),
    Patch(color=ORANGE, label="Decode naive"),
    Patch(color=PURPLE, label="Decode flash"),
    Patch(color=TEAL,   label="Decode flash_v2"),
    Line2D([0],[0], marker="o", color="w", mfc="k", mec="k", ms=8, label="Size = 4k"),
    Line2D([0],[0], marker="s", color="w", mfc="k", mec="k", ms=8, label="Size = 64k"),
    Line2D([0],[0], color="k",    lw=2,    label="Roofline"),
    Line2D([0],[0], color="gray", lw=1.2, ls="--", label=f"Ridge ≈{RIDGE:.1f} FLOPs/B"),
]
ax.legend(handles=legend_handles, fontsize=8.5, loc="lower right",
          framealpha=0.9, edgecolor="#ccc")

# ─── Axes ────────────────────────────────────────────────────────────────────
ax.set_xlabel("Arithmetic Intensity  [FLOPs / byte]", fontsize=12)
ax.set_ylabel("Performance  [GFLOPS]", fontsize=12)
ax.set_title("Roofline — Attention Kernels on NVIDIA TITAN V  (FP32, D=64)", fontsize=13)
ax.set_xlim(0.1, 1e5)
ax.set_ylim(0.7, PEAK_FP32 * 2.5)
ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.55)

os.makedirs("results", exist_ok=True)
plt.tight_layout()
plt.savefig("results/roofline.png", dpi=160)
print("Saved: results/roofline.png")

# ─── Summary table ───────────────────────────────────────────────────────────
print(f"\n{'Label':24s}  {'AI':>9s}  {'GFLOPS':>8s}  {'% of roofline':>14s}")
for label, ai, gf, *_ in points:
    roof = min(PEAK_BW * ai, PEAK_FP32)
    pct  = gf / roof * 100
    tag  = label.replace("\n", " ")
    print(f"{tag:24s}  {ai:>9.1f}  {gf:>8.1f}  {pct:>13.1f}%")