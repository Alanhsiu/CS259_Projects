#!/usr/bin/env python3
"""
plot_roofline.py  –  Roofline chart for CS259 Part 1 (all 6 kernels).
Labels are positioned with explicit offsets + leader arrows to avoid overlap.
"""

import math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = "results/roofline.png"
os.makedirs("results", exist_ok=True)

# ── Hardware (TITAN V, CC 7.0) ─────────────────────────────────────────────
PEAK_FLOPS = 14_900.0
PEAK_BW    =    652.8
RIDGE      = PEAK_FLOPS / PEAK_BW   # ≈ 22.82

TAI_C1 = 142.7
TAI_C2 = 610.2

# ── Measured data ──────────────────────────────────────────────────────────
# (actual_AI, achieved_GFLOPS, marker, color, filled)
DATA = {
    ("conv1","naive")   : (0.141,   86.2,  "o", "#c03820", True ),
    ("conv1","smem_w")  : (0.145,   88.9,  "s", "#c03820", False),
    ("conv1","unroll")  : (0.086,   52.4,  "^", "#c03820", True ),
    ("conv1","smem_wi") : (2.059, 1037.6,  "D", "#c03820", False),
    ("conv1","regblock"): (0.257,  149.0,  "v", "#c03820", True ),
    ("conv1","warp")    : (3.936,  346.1,  "*", "#c03820", True ),
    ("conv2","naive")   : (0.908,  175.7,  "o", "#1a3880", True ),
    ("conv2","smem_w")  : (0.817,  149.2,  "s", "#1a3880", False),
    ("conv2","unroll")  : (0.633,  170.3,  "^", "#1a3880", True ),
    ("conv2","smem_wi") : (87.49, 1181.5,  "D", "#1a3880", False),
    ("conv2","regblock"): (8.258,  631.5,  "v", "#1a3880", True ),
    ("conv2","warp")    : (1.429,  376.8,  "*", "#1a3880", True ),
}

# ── Label offsets (dx, dy) in points from the data point ──────────────────
# Positive dx = right, positive dy = up.
# Use arrows for tightly clustered points.
#
# Cluster 1: conv1 naive (0.141,86) and conv1 smem_w (0.145,89) – nearly identical
# Cluster 2: conv2 unroll (0.633,170), conv2 smem_w (0.817,149), conv2 naive (0.908,176)
LABEL_CFG = {
    # key              : (dx,  dy,   ha,       use_arrow)
    ("conv1","unroll")  : (-58, -18, "right",   False),
    ("conv1","naive")   : (-60,  22, "right",   True ),   # arrow up-left
    ("conv1","smem_w")  : ( 55,  -8, "left",    True ),   # arrow right
    ("conv1","regblock"): (-62,   8, "right",   True ),
    ("conv1","smem_wi") : (  8,   6, "left",    False),
    ("conv1","warp")    : (  8,  -18, "left",   False),   # below smem_wi

    ("conv2","unroll")  : (-62,  20, "right",   True ),   # arrow up-left
    ("conv2","smem_w")  : (  8, -22, "left",    True ),   # arrow down-right
    ("conv2","naive")   : ( 10,  10, "left",    True ),   # arrow up-right
    ("conv2","warp")    : (  8,   6, "left",    False),
    ("conv2","regblock"): (  8,  -18, "left",   False),   # below warp
    ("conv2","smem_wi") : (-125, -22, "right",  True ),   # compute-bound callout
}

ARROW_STYLE = dict(arrowstyle="-", color="#aaaaaa", lw=0.8,
                   shrinkA=4, shrinkB=4)

def logspace(lo, hi, n=600):
    return [10**(lo + (hi - lo) * i / (n - 1)) for i in range(n)]

# ── Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xscale("log")
ax.set_yscale("log")

xs = logspace(-1.2, 3.2)

# Roofline envelope
ax.plot(xs, [min(x * PEAK_BW, PEAK_FLOPS) for x in xs],
        color="#1a6870", lw=2.5, zorder=2, label=f"Mem BW roof  ({PEAK_BW:.0f} GB/s)")
ax.axhline(PEAK_FLOPS, color="#1a3880", lw=2.5, ls="--", zorder=2,
           label=f"FP32 peak  ({PEAK_FLOPS/1e3:.1f} TFLOPS)")

# Ridge
ax.axvline(RIDGE, color="#aaa", lw=1, ls=":")
ax.text(RIDGE * 1.07, PEAK_FLOPS * 0.45, f"Ridge\n{RIDGE:.1f}",
        color="#999", fontsize=8, va="top")

# Theory AI verticals
for tai, lab, col in [
    (TAI_C1, f"Conv1 theory AI = {TAI_C1:.0f}", "#c03820"),
    (TAI_C2, f"Conv2 theory AI = {TAI_C2:.0f}", "#1a3880"),
]:
    ax.axvline(tai, color=col, lw=1.2, ls="--", alpha=0.28)
    ax.text(tai * 1.08, 12, lab, color=col, fontsize=7, alpha=0.55,
            rotation=90, va="bottom")

# ── Scatter points + labels ─────────────────────────────────────────────────
for (layer, kname), (ai, gf, mk, col, filled) in DATA.items():
    # Plot point
    kw = dict(marker=mk, s=140 if mk != "*" else 210, zorder=5, clip_on=False)
    if filled:
        ax.scatter(ai, gf, color=col, **kw)
    else:
        ax.scatter(ai, gf, facecolors="none", edgecolors=col, linewidths=2, **kw)

    # Label
    cfg = LABEL_CFG.get((layer, kname), (8, 5, "left", False))
    dx, dy, ha, use_arrow = cfg
    label_text = f"{layer[-1]}.{kname}"

    ann_kw = dict(
        xy=(ai, gf),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8,
        color=col,
        ha=ha,
        va="center",
        fontweight="semibold",
    )
    if use_arrow:
        ann_kw["arrowprops"] = ARROW_STYLE

    ax.annotate(label_text, **ann_kw)

# ── Axes ────────────────────────────────────────────────────────────────────
ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)", fontsize=11)
ax.set_ylabel("Performance (GFLOPS)", fontsize=11)
ax.set_title(
    "Roofline – NVIDIA TITAN V  |  CS259 Part 1: Convolution  (B=16)",
    fontsize=12,
)
ax.set_xlim(0.055, 1800)
ax.set_ylim(10, PEAK_FLOPS * 2.4)
ax.grid(True, which="both", alpha=0.12)

# ── Legend ──────────────────────────────────────────────────────────────────
kernel_legend = [
    Line2D([0],[0], marker="o", color="k", ls="none", ms=8,
           markerfacecolor="k",    label="K0: naive"),
    Line2D([0],[0], marker="s", color="k", ls="none", ms=8,
           markerfacecolor="none", label="K1: smem\_w"),
    Line2D([0],[0], marker="^", color="k", ls="none", ms=8,
           markerfacecolor="k",    label="K2: unroll"),
    Line2D([0],[0], marker="D", color="k", ls="none", ms=8,
           markerfacecolor="none", label="K3: smem\_wi"),
    Line2D([0],[0], marker="v", color="k", ls="none", ms=8,
           markerfacecolor="k",    label="K4: regblock"),
    Line2D([0],[0], marker="*", color="k", ls="none", ms=10,
           markerfacecolor="k",    label="K5: warp"),
]
layer_legend = [
    Line2D([0],[0], color="#c03820", lw=2.5,
           label="conv1  (224×224, Ni=Nn=64)"),
    Line2D([0],[0], color="#1a3880", lw=2.5,
           label="conv2  (14×14,   Ni=Nn=512)"),
    Line2D([0],[0], color="#1a6870", lw=2.5, label="Mem BW roof"),
    Line2D([0],[0], color="#1a3880", lw=2.5, ls="--", label="Compute roof"),
]
ax.legend(
    handles=kernel_legend + layer_legend,
    fontsize=8.5, loc="lower right", ncol=2, framealpha=0.92,
)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")