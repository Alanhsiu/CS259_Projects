"""
Validation plots for CS259 project2 model.

Outputs in project2/results_<config>:
- model_vs_measured_time.png
- mape_by_kernel.png
"""

from __future__ import annotations

import os
from dataclasses import replace
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import argparse
from matplotlib.lines import Line2D

from hierarchical import (
    KERNELS,
    Sample,
    _fit_kernel_scalers,
    _fit_traffic_correction,
    default_results_dir,
    load_project1_part1_results,
    mape,
    predict_sample,
)


def ensure_results_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "results")
    os.makedirs(out, exist_ok=True)
    return out


def ensure_config_results_dir(config: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, f"results_{config}")
    os.makedirs(out, exist_ok=True)
    return out


def _expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


# ── Roofline constants ──────────────────────────────────────────────────────
PEAK_FLOPS = 14_900.0
PEAK_BW    =    652.8
RIDGE      = PEAK_FLOPS / PEAK_BW

TAI_C1 = 142.7
TAI_C2 = 610.2

MARKERS = {
    "naive": "o",
    "smem_w": "s",
    "unroll": "^",
    "smem_wi": "D",
    "regblock": "v",
    "warp": "*",
}

FILLED = {"naive": True, "smem_w": False, "unroll": True, "smem_wi": False, "regblock": True, "warp": True}

LABEL_CFG = {
    ("conv1","unroll")  : (-58, -18, "right",   False),
    ("conv1","naive")   : (-60,  22, "right",   True ),
    ("conv1","smem_w")  : ( 55,  -8, "left",    True ),
    ("conv1","regblock"): (-62,   8, "right",   True ),
    ("conv1","smem_wi") : (  8,   6, "left",    False),
    ("conv1","warp")    : (  8,  -18, "left",   False),

    ("conv2","unroll")  : (-62,  20, "right",   True ),
    ("conv2","smem_w")  : (  8, -22, "left",    True ),
    ("conv2","naive")   : ( 10,  10, "left",    True ),
    ("conv2","warp")    : (  8,   6, "left",    False),
    ("conv2","regblock"): (  8,  -18, "left",   False),
    ("conv2","smem_wi") : (-125, -22, "right",  True ),
}

ARROW_STYLE = dict(arrowstyle="-", color="#aaaaaa", lw=0.8, shrinkA=4, shrinkB=4)


def logspace(lo, hi, n=600):
    return [10**(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def plot_model_vs_measured(rows, out_dir):
    labels = [f"{r['cfg']}_{r['kernel']}" for r in rows]
    meas = np.array([r["measured_ms"] for r in rows], dtype=float)
    pred = np.array([r["predicted_ms"] for r in rows], dtype=float)

    x = np.arange(len(rows))
    w = 0.4

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(x - w / 2, meas, width=w, color="#1a3880", label="Measured")
    ax.bar(x + w / 2, pred, width=w, color="#1a6870", label="Predicted")

    ax.set_yscale("log")
    ax.set_ylabel("Runtime (ms, log scale)")
    ax.set_title("Measured vs Predicted Runtime (all conv1/conv2 kernels)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    out_path = os.path.join(out_dir, "model_vs_measured_time.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def plot_mape_by_kernel(rows, out_dir):
    grouped = {k: [] for k in KERNELS}
    for r in rows:
        grouped[r["kernel"]].append(r["mape_time_percent"])

    kernels = []
    avg_mapes = []
    for k in KERNELS:
        vals = grouped.get(k, [])
        if vals:
            kernels.append(k)
            avg_mapes.append(sum(vals) / len(vals))

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(kernels, avg_mapes, color="#c03820")
    ax.set_ylabel("Average MAPE (%)")
    ax.set_title("Primary Model Error by Kernel")
    ax.grid(axis="y", alpha=0.3)

    for i, v in enumerate(avg_mapes):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)

    out_path = os.path.join(out_dir, "mape_by_kernel.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def plot_roofline_predicted(rows, out_dir):
    """Generate roofline plot from predicted results."""
    data = {}
    for r in rows:
        key = (r["cfg"], r["kernel"])
        ai = r.get("ai_dram", None)
        tflops = r.get("tflops", None)
        if ai is None or tflops is None:
            continue
        gf = tflops * 1e3  # TFLOPS -> GFLOPS
        col = "#c03820" if r["cfg"] == "conv1" else "#1a3880"
        mk = MARKERS.get(r["kernel"], "o")
        filled = FILLED.get(r["kernel"], True)
        data[key] = (ai, gf, mk, col, filled)

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
    ax.text(RIDGE * 1.07, PEAK_FLOPS * 0.45, f"Ridge\n{RIDGE:.1f}", color="#999", fontsize=8, va="top")

    # Theory AI verticals
    for tai, lab, col in [
        (TAI_C1, f"Conv1 theory AI = {TAI_C1:.0f}", "#c03820"),
        (TAI_C2, f"Conv2 theory AI = {TAI_C2:.0f}", "#1a3880"),
    ]:
        ax.axvline(tai, color=col, lw=1.2, ls="--", alpha=0.28)
        ax.text(tai * 1.08, 12, lab, color=col, fontsize=7, alpha=0.55,
                rotation=90, va="bottom")

    # Scatter points + labels
    for (layer, kname), (ai, gf, mk, col, filled) in data.items():
        kw = dict(marker=mk, s=140 if mk != "*" else 210, zorder=5, clip_on=False)
        if filled:
            ax.scatter(ai, gf, color=col, **kw)
        else:
            ax.scatter(ai, gf, facecolors="none", edgecolors=col, linewidths=2, **kw)

        cfg = LABEL_CFG.get((layer, kname), (8, 5, "left", False))
        dx, dy, ha, use_arrow = cfg
        label_text = f"{layer[-1]}.{kname}"
        ann_kw = dict(xy=(ai, gf), xytext=(dx, dy), textcoords="offset points",
                      fontsize=8, color=col, ha=ha, va="center", fontweight="semibold")
        if use_arrow:
            ann_kw["arrowprops"] = ARROW_STYLE
        ax.annotate(label_text, **ann_kw)

    # Axes
    ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)", fontsize=11)
    ax.set_ylabel("Performance (GFLOPS)", fontsize=11)
    ax.set_title("Roofline – Predicted (hierarchical model)  |  NVIDIA TITAN V  (B=16)", fontsize=12)
    ax.set_xlim(0.055, 1800)
    ax.set_ylim(10, PEAK_FLOPS * 2.4)
    ax.grid(True, which="both", alpha=0.12)

    kernel_legend = [
        Line2D([0],[0], marker="o", color="k", ls="none", ms=8, markerfacecolor="k",    label="K0: naive"),
        Line2D([0],[0], marker="s", color="k", ls="none", ms=8, markerfacecolor="none", label="K1: smem_w"),
        Line2D([0],[0], marker="^", color="k", ls="none", ms=8, markerfacecolor="k",    label="K2: unroll"),
        Line2D([0],[0], marker="D", color="k", ls="none", ms=8, markerfacecolor="none", label="K3: smem_wi"),
        Line2D([0],[0], marker="v", color="k", ls="none", ms=8, markerfacecolor="k",    label="K4: regblock"),
        Line2D([0],[0], marker="*", color="k", ls="none", ms=10, markerfacecolor="k",    label="K5: warp"),
    ]
    layer_legend = [
        Line2D([0],[0], color="#c03820", lw=2.5, label="conv1  (224×224, Ni=Nn=64)"),
        Line2D([0],[0], color="#1a3880", lw=2.5, label="conv2  (14×14,   Ni=Nn=512)"),
        Line2D([0],[0], color="#1a6870", lw=2.5, label="Mem BW roof"),
        Line2D([0],[0], color="#1a3880", lw=2.5, ls="--", label="Compute roof"),
    ]
    ax.legend(handles=kernel_legend + layer_legend, fontsize=8.5, loc="lower right", ncol=2, framealpha=0.92)

    out_path = os.path.join(out_dir, "roofline_predicted.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def _remap_samples(samples, ky: int, kx: int):
    if ky == 3 and kx == 3:
        return samples
    return [replace(s, ky=ky, kx=kx) for s in samples]


def evaluate_for_validation(results_dir: str, ky: int, kx: int):
    samples = _remap_samples(load_project1_part1_results(results_dir), ky=ky, kx=kx)
    traffic_fit = _fit_traffic_correction(samples)
    params_primary = _fit_kernel_scalers(samples, use_measured_dram=False, traffic_fit=traffic_fit)

    primary_rows = []

    for s in sorted(samples, key=lambda x: (x.cfg, KERNELS.index(x.kernel))):
        pred_primary = predict_sample(s, params_primary, traffic_fit=traffic_fit, use_measured_dram=False)

        measured_tflops = s.gflops / 1000.0
        measured_ai = s.flops / s.dram_bytes
        measured_dram_gb = s.dram_bytes / 1e9

        primary_rows.append(
            {
                "cfg": s.cfg,
                "kernel": s.kernel,
                "measured_ms": s.time_ms,
                "predicted_ms": pred_primary["predicted_time_ms"],
                "tflops": pred_primary["tflops"],
                "mape_time_percent": mape(pred_primary["predicted_time_ms"], s.time_ms),
                "mape_tflops_percent": mape(pred_primary["tflops"], measured_tflops),
                "mape_ai_percent": mape(pred_primary["ai_dram"], measured_ai),
                "dram_GB_used": pred_primary["dram_bytes_GB"],
                "mape_dram_percent": mape(pred_primary["dram_bytes_GB"], measured_dram_gb),
                "ai_dram": pred_primary["ai_dram"],
                "measured_tflops": measured_tflops,
                "measured_ai": measured_ai,
                "ideal_compute_ms": pred_primary["ideal_compute_ms"],
                "ideal_mem_ms": pred_primary["ideal_mem_ms"],
            }
        )

    return primary_rows


def main():
    parser = argparse.ArgumentParser(description="Validate the hierarchical model on a results directory.")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Path to a directory containing summary.txt and per-kernel metrics files.",
    )
    parser.add_argument(
        "--config",
        default="default",
        help="Config name used to build the output directory results_<config>.",
    )
    parser.add_argument(
        "--input-root",
        default=None,
        help="Root directory containing the measured results tree (defaults to project1/part1 or the copied dataset).",
    )
    parser.add_argument("--ky", type=int, default=3, help="Kernel height used when validating the model.")
    parser.add_argument("--kx", type=int, default=3, help="Kernel width used when validating the model.")
    args = parser.parse_args()

    if args.results_dir is not None:
        results_dir = _expand_path(args.results_dir)
    elif args.input_root is not None:
        results_dir = _expand_path(args.input_root)
    else:
        results_dir = default_results_dir()

    out_dir = ensure_config_results_dir(args.config)
    primary_rows = evaluate_for_validation(results_dir, ky=args.ky, kx=args.kx)

    plot_model_vs_measured(primary_rows, out_dir)
    plot_mape_by_kernel(primary_rows, out_dir)
    plot_roofline_predicted(primary_rows, out_dir)

    avg_mape_primary = sum(r["mape_time_percent"] for r in primary_rows) / len(primary_rows)
    avg_mape_dram = sum(r.get("mape_dram_percent", 0.0) for r in primary_rows) / len(primary_rows)

    print("\nSummary:")
    print(f"  Average MAPE (primary analytic model): {avg_mape_primary:.2f}%")
    print(f"  Average DRAM MAPE (primary analytic model): {avg_mape_dram:.2f}%")
    print(f"  Results(input) directory: {results_dir}")
    print(f"  Output directory: {out_dir}")


if __name__ == "__main__":
    main()
