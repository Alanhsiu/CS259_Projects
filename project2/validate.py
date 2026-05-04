"""
Validation plots for CS259 project2 model.

Outputs in project2/results:
- model_vs_measured_time.png
- mape_by_kernel.png
- measured_vs_analytic_dram.png
"""

from __future__ import annotations

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import argparse

from hierarchical import KERNELS, default_results_dir, evaluate


def ensure_results_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "results")
    os.makedirs(out, exist_ok=True)
    return out


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


def plot_measured_vs_analytic_mode(measured_rows, analytic_rows, out_dir):
    labels = [f"{r['cfg']}_{r['kernel']}" for r in measured_rows]
    mape_meas = np.array([r["mape_time_percent"] for r in measured_rows], dtype=float)
    mape_ana = np.array([r["mape_time_percent"] for r in analytic_rows], dtype=float)

    x = np.arange(len(labels))
    w = 0.4

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(x - w / 2, mape_meas, width=w, color="#1a6870", label="Shortcut baseline")
    ax.bar(x + w / 2, mape_ana, width=w, color="#8B4513", label="Primary analytic model")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Error comparison: shortcut baseline vs primary analytic model")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    out_path = os.path.join(out_dir, "measured_vs_analytic_dram.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out_path}")


def main():
    out_dir = ensure_results_dir()
    data = evaluate(default_results_dir())

    shortcut_rows = data["shortcut_mode_rows"]
    primary_rows = data["primary_mode_rows"]

    plot_model_vs_measured(primary_rows, out_dir)
    plot_mape_by_kernel(primary_rows, out_dir)
    plot_measured_vs_analytic_mode(shortcut_rows, primary_rows, out_dir)

    avg_mape_shortcut = sum(r["mape_time_percent"] for r in shortcut_rows) / len(shortcut_rows)
    avg_mape_primary = sum(r["mape_time_percent"] for r in primary_rows) / len(primary_rows)

    print("\nSummary:")
    print(f"  Average MAPE (shortcut baseline): {avg_mape_shortcut:.2f}%")
    print(f"  Average MAPE (primary analytic model): {avg_mape_primary:.2f}%")


if __name__ == "__main__":
    main()
