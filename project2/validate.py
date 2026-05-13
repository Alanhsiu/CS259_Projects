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
    predict_smwi_tile_sweep,
    predict_smwi_problem_sweep,
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


def plot_error_vs_tile_size(measured_csv_path: str, out_dir: str, results_dir: str = None):
    """Plot measured vs model vs roofline for tile size sweep with per-point MAPE.

    Top panel: absolute runtimes.
    Bottom panel: MAPE bars (hierarchical model vs roofline baseline).
    """
    import csv
    from roofline import roofline as roofline_predict

    if not os.path.exists(measured_csv_path):
        print(f"Warning: {measured_csv_path} not found, skipping tile size plot.")
        return

    tile_nis, measured_times = [], []
    try:
        with open(measured_csv_path, "r") as f:
            for row in csv.DictReader(f):
                tile_nis.append(int(row["smwi_tile_ni"]))
                measured_times.append(float(row["time_ms"]))
    except Exception as e:
        print(f"Error reading {measured_csv_path}: {e}")
        return

    if not tile_nis:
        return

    pairs = sorted(zip(tile_nis, measured_times))
    tile_nis = [x[0] for x in pairs]
    measured_times = [x[1] for x in pairs]

    # Hierarchical model predictions
    preds = predict_smwi_tile_sweep(
        B=16, Ny=224, Nx=224, Ni=64, Nn=64,
        tile_ni_list=tile_nis,
        results_dir=results_dir,
    )
    predicted_times = [p["predicted_ms"] for p in preds]

    # Roofline baseline (constant — doesn't depend on tile_ni)
    rf = roofline_predict(16, 224, 224, 64, 64)
    roofline_time = rf["predicted_time_ms"]

    model_mapes = [abs(p - m) / m * 100 for p, m in zip(predicted_times, measured_times)]
    roof_mapes  = [abs(roofline_time - m) / m * 100 for m in measured_times]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))

    # ── top: runtime ───────────────────────────────────────────────────────
    ax1.plot(tile_nis, measured_times,  "o-",  color="#c03820", lw=2, ms=8, label="Measured")
    ax1.plot(tile_nis, predicted_times, "s--", color="#1a3880", lw=2, ms=8, label="Hierarchical model")
    ax1.axhline(roofline_time, color="#1a6870", lw=1.5, ls=":",
                label=f"Roofline ({roofline_time:.1f} ms)")
    ax1.set_ylabel("Runtime (ms)", fontsize=11)
    ax1.set_title("K3 smem_wi: Runtime vs Tile Size  |  Conv1 224×224 (Ni=Nn=64)", fontsize=12)
    ax1.set_xticks(tile_nis)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # ── bottom: MAPE bars ──────────────────────────────────────────────────
    x = np.arange(len(tile_nis))
    w = 0.35
    ax2.bar(x - w / 2, model_mapes, width=w, color="#1a3880", label="Hierarchical model")
    ax2.bar(x + w / 2, roof_mapes,  width=w, color="#aaaaaa",  label="Roofline baseline")
    for i, (m, r) in enumerate(zip(model_mapes, roof_mapes)):
        ax2.text(i - w / 2, m + 0.3, f"{m:.1f}%", ha="center", fontsize=8)
        ax2.text(i + w / 2, r + 0.3, f"{r:.1f}%", ha="center", fontsize=8)
    ax2.set_xlabel("SMWI_TILE_NI (tile size)", fontsize=11)
    ax2.set_ylabel("MAPE (%)", fontsize=11)
    ax2.set_title("Prediction Error vs Tile Size", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tile_nis)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(fontsize=10)

    avg_m = sum(model_mapes) / len(model_mapes)
    avg_r = sum(roof_mapes) / len(roof_mapes)
    print(f"  Tile sweep — model avg MAPE: {avg_m:.1f}%  roofline avg MAPE: {avg_r:.1f}%")

    out_path = os.path.join(out_dir, "error_vs_tile_size.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def plot_error_vs_problem_size(measured_csv_path: str, out_dir: str, results_dir: str = None):
    """Plot measured vs model vs roofline for problem size sweep with per-point MAPE.

    Top panel: absolute runtimes.
    Bottom panel: MAPE bars comparing hierarchical model to roofline baseline.
    """
    import csv
    from roofline import roofline as roofline_predict

    if not os.path.exists(measured_csv_path):
        print(f"Warning: {measured_csv_path} not found, skipping problem size plot.")
        return

    problem_sizes, measured_times = [], []
    try:
        with open(measured_csv_path, "r") as f:
            for row in csv.DictReader(f):
                problem_sizes.append(int(row["Ny"]))
                measured_times.append(float(row["time_ms"]))
    except Exception as e:
        print(f"Error reading {measured_csv_path}: {e}")
        return

    if not problem_sizes:
        return

    pairs = sorted(zip(problem_sizes, measured_times))
    problem_sizes = [x[0] for x in pairs]
    measured_times = [x[1] for x in pairs]

    # Hierarchical model predictions for each (Ny, Nx) = (Ny, Ny)
    preds = predict_smwi_problem_sweep(
        Ny_list=problem_sizes, B=16, Ni=64, Nn=64,
        results_dir=results_dir,
    )
    predicted_times = [p["predicted_ms"] for p in preds]

    # Roofline baseline (scales with Ny^2 — proportional to FLOPs/bytes)
    roofline_times = [roofline_predict(16, Ny, Ny, 64, 64)["predicted_time_ms"]
                      for Ny in problem_sizes]

    model_mapes = [abs(p - m) / m * 100 for p, m in zip(predicted_times, measured_times)]
    roof_mapes  = [abs(r - m) / m * 100 for r, m in zip(roofline_times,  measured_times)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))

    # ── top: runtime ───────────────────────────────────────────────────────
    ax1.plot(problem_sizes, measured_times,  "o-",  color="#1a3880", lw=2, ms=8, label="Measured")
    ax1.plot(problem_sizes, predicted_times, "s--", color="#c03820", lw=2, ms=8, label="Hierarchical model")
    ax1.plot(problem_sizes, roofline_times,  "^:",  color="#1a6870", lw=1.5, ms=8, label="Roofline baseline")
    ax1.set_ylabel("Runtime (ms)", fontsize=11)
    ax1.set_title("K3 smem_wi: Runtime vs Problem Size  |  Conv1 (Ni=Nn=64, B=16)", fontsize=12)
    ax1.set_yscale("log")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=10)

    # ── bottom: MAPE bars ──────────────────────────────────────────────────
    x = np.arange(len(problem_sizes))
    w = 0.35
    ax2.bar(x - w / 2, model_mapes, width=w, color="#c03820", label="Hierarchical model")
    ax2.bar(x + w / 2, roof_mapes,  width=w, color="#aaaaaa",  label="Roofline baseline")
    for i, (m, r) in enumerate(zip(model_mapes, roof_mapes)):
        ax2.text(i - w / 2, m + 0.3, f"{m:.1f}%", ha="center", fontsize=8)
        ax2.text(i + w / 2, r + 0.3, f"{r:.1f}%", ha="center", fontsize=8)
    ax2.set_xlabel("Problem Size (Ny = Nx)", fontsize=11)
    ax2.set_ylabel("MAPE (%)", fontsize=11)
    ax2.set_title("Prediction Error vs Problem Size", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(problem_sizes)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(fontsize=10)

    avg_m = sum(model_mapes) / len(model_mapes)
    avg_r = sum(roof_mapes) / len(roof_mapes)
    print(f"  Problem sweep — model avg MAPE: {avg_m:.1f}%  roofline avg MAPE: {avg_r:.1f}%")

    out_path = os.path.join(out_dir, "error_vs_problem_size.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")



def plot_sensitivity_analysis(out_dir: str, results_dir: str = None):
    """Sensitivity analysis: how much does predicted time change when HW params vary?

    For each parameter (DRAM BW, FP32 TFLOPS, L2 size), we sweep a multiplier
    range and recompute the predicted time for smem_wi at Conv1 and Conv2.

    This answers: "Which hardware parameters should a future GPU change to most
    improve performance on this kernel?"
    """
    from hierarchical import (
        HW, FP32_TFLOPS, DRAM_BW_GB_S, evaluate, _corrected_analytic_dram_bytes,
        conv_flops, Sample,
    )

    if results_dir is None:
        results_dir = default_results_dir()

    fit = evaluate(results_dir)
    params = fit["kernel_params_primary"]["smem_wi"]
    traffic_fit = fit["traffic_fit"]
    sc = params["scale_compute"]
    sm = params["scale_mem"]

    configs = [
        ("Conv1\n224×224, Ni=Nn=64", 16, 224, 224, 64, 64),
        ("Conv2\n14×14, Ni=Nn=512",  16,  14,  14, 512, 512),
    ]

    multipliers = np.linspace(0.25, 4.0, 60)
    sweep_params = [
        ("DRAM Bandwidth",     "dram_bw_gb_s",  "×  base = 652.8 GB/s",  "#c03820"),
        ("FP32 Peak TFLOPS",   "fp32_tflops",   "×  base = 14.9 TFLOPS", "#1a3880"),
        ("L2 Cache Size",      "l2_bytes",      "×  base = 4.5 MB",       "#1a6870"),
    ]

    for param_name, param_key, x_label, color in sweep_params:
        fig, axes = plt.subplots(1, len(configs), figsize=(13, 5), sharey=False)
        fig.suptitle(f"Sensitivity: {param_name}  |  K3 smem_wi", fontsize=13)

        for ax, (cfg_name, B, Ny, Nx, Ni, Nn) in zip(axes, configs):
            sample = Sample(
                cfg="sens", kernel="smem_wi", B=B, Ny=Ny, Nx=Nx, Ni=Ni, Nn=Nn,
                ky=3, kx=3, groups=1, time_ms=0.0, gflops=0.0,
                dram_read_bytes=0.0, dram_write_bytes=0.0,
            )
            base_dram = _corrected_analytic_dram_bytes(sample, traffic_fit)
            base_flops = conv_flops(B, Ny, Nx, Ni, Nn, 3, 3)
            base_bw = DRAM_BW_GB_S * 1e9
            base_fp32 = FP32_TFLOPS * 1e12
            base_l2 = float(HW["l2_bytes"])

            syn_b = 3 * 3 * Nn * Ni * 4

            times = []
            for mult in multipliers:
                if param_key == "dram_bw_gb_s":
                    eff_bw = base_bw * mult
                    t_comp = base_flops / base_fp32 * 1e3
                    t_mem  = base_dram / eff_bw * 1e3
                elif param_key == "fp32_tflops":
                    t_comp = base_flops / (base_fp32 * mult) * 1e3
                    t_mem  = base_dram / base_bw * 1e3
                else:  # l2_bytes
                    eff_l2 = base_l2 * mult
                    # Smaller L2 → weights evicted more → more DRAM traffic
                    if syn_b < eff_l2:
                        dram = base_dram  # weights cache → no extra traffic
                    else:
                        dram_extra = syn_b - base_dram * 0.01  # proxy for extra weight fetch
                        dram = base_dram + max(0, dram_extra)
                    t_comp = base_flops / base_fp32 * 1e3
                    t_mem  = dram / base_bw * 1e3
                times.append(max(sc * t_comp, sm * t_mem))

            baseline_time = times[len(multipliers) // 4]  # at mult=1.0 approx
            # Find the index closest to mult=1.0
            idx1 = np.argmin(np.abs(multipliers - 1.0))
            baseline_time = times[idx1]

            ax.plot(multipliers, times, color=color, lw=2.5)
            ax.axvline(1.0, color="#999", lw=1, ls="--")
            ax.scatter([1.0], [baseline_time], color=color, s=80, zorder=5)
            ax.set_xlabel(f"Multiplier  ({x_label})", fontsize=10)
            ax.set_ylabel("Predicted time (ms)", fontsize=10)
            ax.set_title(cfg_name, fontsize=11)
            ax.grid(True, alpha=0.2)

        fname = f"sensitivity_{param_key}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {os.path.join(out_dir, fname)}")


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
    
    # Plot sweep results if available
    measured_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "measured")
    plot_error_vs_tile_size(
        os.path.join(measured_dir, "tile_size_sweep.csv"), out_dir, results_dir=results_dir
    )
    plot_error_vs_problem_size(
        os.path.join(measured_dir, "problem_size_sweep.csv"), out_dir, results_dir=results_dir
    )

    # Sensitivity analysis
    plot_sensitivity_analysis(out_dir, results_dir=results_dir)

    avg_mape_primary = sum(r["mape_time_percent"] for r in primary_rows) / len(primary_rows)
    avg_mape_dram = sum(r.get("mape_dram_percent", 0.0) for r in primary_rows) / len(primary_rows)

    print("\nSummary:")
    print(f"  Average MAPE (primary analytic model): {avg_mape_primary:.2f}%")
    print(f"  Average DRAM MAPE (primary analytic model): {avg_mape_dram:.2f}%")
    print(f"  Results(input) directory: {results_dir}")
    print(f"  Output directory: {out_dir}")


if __name__ == "__main__":
    main()
