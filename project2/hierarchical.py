"""
Calibrated hierarchical model for CS259 project2.

What this script does:
1) Loads measured timings from project1 part1 summary + ncu metrics.
2) Builds a mechanistic analytic DRAM model for each kernel.
3) Fits a small shape-aware correction from Conv1/Conv2 measurements so the
    analytic model is more universal than a pure measured-bytes shortcut.
4) Keeps a measured-DRAM shortcut baseline for comparison.
5) Writes CSV tables for report use.
"""

from __future__ import annotations

import csv
import math
import os
import re
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional, Tuple

from hw_params import HW


def conv_flops(B: int, Ny: int, Nx: int, Ni: int, Nn: int, ky: int, kx: int, groups: int = 1) -> float:
    cin_per_group = Ni // groups
    return float(2 * B * Ny * Nx * Nn * ky * kx * cin_per_group)


def conv_tensor_bytes(
    B: int,
    Ny: int,
    Nx: int,
    Ni: int,
    Nn: int,
    ky: int,
    kx: int,
    groups: int = 1,
    dtype_bytes: int = 4,
) -> Tuple[float, float, float]:
    cin_per_group = Ni // groups
    nypad = Ny + ky - 1
    nxpad = Nx + kx - 1
    syn_bytes = ky * kx * Nn * cin_per_group * dtype_bytes
    inp_bytes = B * nypad * nxpad * Ni * dtype_bytes
    out_bytes = B * Ny * Nx * Nn * dtype_bytes
    return float(syn_bytes), float(inp_bytes), float(out_bytes)


def conv_l2_pressure(B: int, Ny: int, Nx: int, Ni: int, Nn: int, ky: int, kx: int, groups: int = 1) -> float:
    """A simple shape feature for traffic correction.

    It is not a hardware simulator. It only captures how large the biggest
    tensor footprint is relative to L2.
    """
    syn_b, inp_b, out_b = conv_tensor_bytes(B, Ny, Nx, Ni, Nn, ky, kx, groups)
    return max(syn_b, inp_b, out_b) / HW["l2_bytes"]

# Known configs from project1/part1
CONFIGS = {
    "conv1": {"B": 16, "Ny": 224, "Nx": 224, "Ni": 64, "Nn": 64, "ky": 3, "kx": 3, "groups": 1},
    "conv2": {"B": 16, "Ny": 14, "Nx": 14, "Ni": 512, "Nn": 512, "ky": 3, "kx": 3, "groups": 1},
}

KERNELS = ["naive", "smem_w", "unroll", "smem_wi", "regblock", "warp"]

SUMMARY_RE = re.compile(
    r"^\s*(naive|smem_w|unroll|smem_wi|regblock|warp)\s+"
    r"([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)

METRIC_PATTERNS = {
    "dram_read": re.compile(r"dram__bytes_read\.sum\s+(Gbyte|Mbyte)\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    "dram_write": re.compile(r"dram__bytes_write\.sum\s+(Gbyte|Mbyte)\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
}


@dataclass
class Sample:
    cfg: str
    kernel: str
    B: int
    Ny: int
    Nx: int
    Ni: int
    Nn: int
    ky: int
    kx: int
    groups: int
    time_ms: float
    gflops: float
    dram_read_bytes: float
    dram_write_bytes: float

    @property
    def dram_bytes(self) -> float:
        return self.dram_read_bytes + self.dram_write_bytes

    @property
    def flops(self) -> float:
        return conv_flops(self.B, self.Ny, self.Nx, self.Ni, self.Nn, self.ky, self.kx, self.groups)


def _to_bytes(unit: str, value: float) -> float:
    if unit.lower().startswith("g"):
        return value * 1e9
    if unit.lower().startswith("m"):
        return value * 1e6
    raise ValueError(f"Unsupported unit: {unit}")


def _parse_summary(summary_path: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    rows: Dict[Tuple[str, str], Dict[str, float]] = {}
    cur_cfg: Optional[str] = None

    with open(summary_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            low = line.lower()
            if "=== conv1-vgg" in low:
                cur_cfg = "conv1"
                continue
            if "=== conv2-vgg" in low:
                cur_cfg = "conv2"
                continue
            m = SUMMARY_RE.search(line)
            if not m or cur_cfg is None:
                continue
            kernel = m.group(1).lower()
            time_ms = float(m.group(2))
            gflops = float(m.group(3))
            rows[(cur_cfg, kernel)] = {
                "time_ms": time_ms,
                "gflops": gflops,
            }

    return rows


def _parse_metrics(metrics_path: str) -> Tuple[float, float]:
    with open(metrics_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    read_m = METRIC_PATTERNS["dram_read"].search(text)
    write_m = METRIC_PATTERNS["dram_write"].search(text)
    if not read_m or not write_m:
        raise ValueError(f"Missing DRAM metric(s) in {metrics_path}")

    read_bytes = _to_bytes(read_m.group(1), float(read_m.group(2)))
    write_bytes = _to_bytes(write_m.group(1), float(write_m.group(2)))
    return read_bytes, write_bytes


def load_project1_part1_results(results_dir: str) -> List[Sample]:
    summary_path = os.path.join(results_dir, "summary.txt")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Cannot find summary: {summary_path}")

    summary_rows = _parse_summary(summary_path)
    samples: List[Sample] = []

    for cfg in ["conv1", "conv2"]:
        for kernel in KERNELS:
            key = (cfg, kernel)
            if key not in summary_rows:
                continue
            metrics_path = os.path.join(results_dir, f"{cfg}_{kernel}_metrics.txt")
            read_bytes, write_bytes = _parse_metrics(metrics_path)
            p = CONFIGS[cfg]
            samples.append(
                Sample(
                    cfg=cfg,
                    kernel=kernel,
                    B=p["B"],
                    Ny=p["Ny"],
                    Nx=p["Nx"],
                    Ni=p["Ni"],
                    Nn=p["Nn"],
                    ky=p["ky"],
                    kx=p["kx"],
                    groups=p.get("groups", 1),
                    time_ms=summary_rows[key]["time_ms"],
                    gflops=summary_rows[key]["gflops"],
                    dram_read_bytes=read_bytes,
                    dram_write_bytes=write_bytes,
                )
            )

    if not samples:
        raise RuntimeError(f"No samples parsed from {results_dir}")

    return samples


def _analytic_dram_bytes(sample: Sample, smwi_tile_ni: int = 8, reg_n: int = 4) -> float:
    """Analytic DRAM traffic estimate for each kernel.

    This is intentionally simple but mechanistic: it tracks tensor traversal,
    tiling reuse, and coalescing assumptions.
    """
    B, Ny, Nx, Ni, Nn = sample.B, sample.Ny, sample.Nx, sample.Ni, sample.Nn
    ky, kx, groups = sample.ky, sample.kx, sample.groups
    cin_per_group = Ni // groups
    dtype_bytes = 4
    syn_b, inp_b, out_b = conv_tensor_bytes(B, Ny, Nx, Ni, Nn, ky, kx, groups, dtype_bytes)

    if sample.kernel == "naive":
        input_dram = B * Ny * Nx * Nn * ky * kx * cin_per_group * dtype_bytes
        return input_dram + syn_b + out_b

    if sample.kernel == "smem_w":
        input_dram = B * Ny * Nx * Nn * ky * kx * cin_per_group * dtype_bytes
        return input_dram + syn_b + out_b

    if sample.kernel == "unroll":
        # Extra instruction/index overhead often increases memory-side stalls.
        input_dram = 1.25 * (B * Ny * Nx * Nn * ky * kx * cin_per_group * dtype_bytes)
        return input_dram + syn_b + out_b

    if sample.kernel == "smem_wi":
        tile_x = 16
        tile_y = 16
        inp_tile_x = tile_x + kx - 1
        inp_tile_y = tile_y + ky - 1

        num_tile_x = math.ceil(Nx / tile_x)
        num_tile_y = math.ceil(Ny / tile_y)
        num_tile_ni = math.ceil(cin_per_group / smwi_tile_ni)

        inp_patch = inp_tile_x * inp_tile_y * smwi_tile_ni * dtype_bytes
        input_dram = B * Nn * num_tile_x * num_tile_y * num_tile_ni * inp_patch

        w_chunk = ky * kx * smwi_tile_ni * dtype_bytes
        weight_dram = B * Nn * num_tile_x * num_tile_y * num_tile_ni * w_chunk

        # If weights fit in L2, cap to a few full sweeps.
        if syn_b < HW["l2_bytes"]:
            weight_dram = max(syn_b, syn_b * (B * Nn / HW["num_sms"]))

        return input_dram + weight_dram + out_b

    if sample.kernel == "regblock":
        input_dram = (B * Ny * Nx * Nn * ky * kx * cin_per_group * dtype_bytes) / reg_n
        return input_dram + syn_b + out_b

    if sample.kernel == "warp":
        cache_line = 128
        rounds = math.ceil(ky * kx * cin_per_group / HW["warp_size"])
        input_lines = B * Ny * Nx * Nn * rounds
        input_dram = input_lines * cache_line
        return input_dram + syn_b + out_b

    raise ValueError(f"Unknown kernel: {sample.kernel}")


def _fit_traffic_correction(rows: List[Sample]) -> Dict[str, Dict[str, float]]:
    """Fit a small log-linear correction to analytic DRAM traffic.

    Model form per kernel:
      corrected_dram = analytic_dram * exp(a + b * log(l2_pressure))
    where l2_pressure = max(tensor_footprint) / L2_size.

    This keeps the model analytic-first, but lets Conv1/Conv2 data correct the
    systematic traffic gap in a shape-aware way.
    """
    by_kernel: Dict[str, List[Sample]] = {}
    for r in rows:
        by_kernel.setdefault(r.kernel, []).append(r)

    fits: Dict[str, Dict[str, float]] = {}
    for kernel, kres in by_kernel.items():
        xs: List[float] = []
        ys: List[float] = []
        for r in kres:
            analytic_bytes = _analytic_dram_bytes(r)
            pressure = max(conv_l2_pressure(r.B, r.Ny, r.Nx, r.Ni, r.Nn, r.ky, r.kx, r.groups), 1e-6)
            ratio = max(r.dram_bytes / analytic_bytes, 1e-12)
            xs.append(math.log(pressure))
            ys.append(math.log(ratio))

        if len(xs) == 1:
            slope = 0.0
            intercept = ys[0]
        else:
            x_mean = mean(xs)
            y_mean = mean(ys)
            denom = sum((x - x_mean) ** 2 for x in xs)
            slope = 0.0 if abs(denom) < 1e-12 else sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
            intercept = y_mean - slope * x_mean

        fits[kernel] = {
            "traffic_intercept": intercept,
            "traffic_slope": slope,
        }

    return fits


def _corrected_analytic_dram_bytes(sample: Sample, traffic_fit: Dict[str, Dict[str, float]]) -> float:
    analytic_bytes = _analytic_dram_bytes(sample)
    pressure = max(conv_l2_pressure(sample.B, sample.Ny, sample.Nx, sample.Ni, sample.Nn, sample.ky, sample.kx, sample.groups), 1e-6)
    fit = traffic_fit[sample.kernel]
    multiplier = math.exp(fit["traffic_intercept"] + fit["traffic_slope"] * math.log(pressure))
    # Prevent pathological over-correction outside the fitted regime.
    multiplier = min(max(multiplier, 0.05), 20.0)
    return analytic_bytes * multiplier


def mape(pred: float, meas: float) -> float:
    return abs(pred - meas) / meas * 100.0


def _ideal_times_ms(flops: float, dram_bytes: float) -> Tuple[float, float]:
    t_compute_ms = flops / (HW["fp32_tflops"] * 1e12) * 1e3
    t_mem_ms = dram_bytes / (HW["dram_bw_gb_s"] * 1e9) * 1e3
    return t_compute_ms, t_mem_ms


def _fit_kernel_scalers(rows: List[Sample]) -> Dict[str, Dict[str, float]]:
    """Fit per-kernel scale factors for compute and DRAM ideal times.

    Model form:
      pred_ms = max(scale_compute * ideal_compute_ms,
                    scale_mem * ideal_mem_ms)
    """
    by_kernel: Dict[str, List[Sample]] = {}
    for r in rows:
        by_kernel.setdefault(r.kernel, []).append(r)

    params: Dict[str, Dict[str, float]] = {}

    search = [0.5, 0.75, 1.0]
    search += [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0]

    for kernel, kres in by_kernel.items():
        best = {"scale_compute": 1.0, "scale_mem": 1.0, "mape": float("inf")}
        for sc in search:
            for sm in search:
                errs = []
                for r in kres:
                    c_ms, m_ms = _ideal_times_ms(r.flops, r.dram_bytes)
                    pred = max(sc * c_ms, sm * m_ms)
                    errs.append(mape(pred, r.time_ms))
                avg_err = mean(errs)
                if avg_err < best["mape"]:
                    best = {"scale_compute": sc, "scale_mem": sm, "mape": avg_err}
        params[kernel] = best

    return params


def predict_sample(
    sample: Sample,
    kernel_params: Dict[str, Dict[str, float]],
    traffic_fit: Optional[Dict[str, Dict[str, float]]] = None,
    use_measured_dram: bool = True,
) -> Dict[str, float]:
    if use_measured_dram:
        dram_bytes = sample.dram_bytes
    else:
        if traffic_fit is None:
            raise ValueError("traffic_fit is required for analytic predictions")
        dram_bytes = _corrected_analytic_dram_bytes(sample, traffic_fit)
    c_ms, m_ms = _ideal_times_ms(sample.flops, dram_bytes)

    p = kernel_params[sample.kernel]
    pred_ms = max(p["scale_compute"] * c_ms, p["scale_mem"] * m_ms)

    # Calculate TFLOPS (Tera operations per second)
    tflops = sample.flops / (pred_ms * 1e9)

    return {
        "predicted_time_ms": pred_ms,
        "tflops": tflops,
        "ai_dram": sample.flops / dram_bytes,
        "ideal_compute_ms": c_ms,
        "ideal_mem_ms": m_ms,
        "dram_bytes_GB": dram_bytes / 1e9,
        "kernel": sample.kernel,
        "cfg": sample.cfg,
    }


def write_predictions_csv(path: str, rows: List[Dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "cfg",
        "kernel",
        "measured_ms",
        "predicted_ms",
        "tflops",
        "mape_time_percent",
        "mape_tflops_percent",
        "mape_ai_percent",
        "dram_GB_used",
        "measured_tflops",
        "measured_ai",
        "ai_dram",
        "ideal_compute_ms",
        "ideal_mem_ms",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _print_table(title: str, table: List[Dict[str, float]]) -> None:
    print("\n" + title)
    print("-" * len(title))
    print(f"{'cfg':<6} {'kernel':<9} {'Time MAPE%':>10} {'TFLOPS Err%':>12} {'AI MAPE%':>10}")
    for r in table:
        print(
            f"{r['cfg']:<6} {r['kernel']:<9} {r['mape_time_percent']:>10.2f} "
            f"{r['mape_tflops_percent']:>12.2f} {r['mape_ai_percent']:>10.2f}"
        )
    avg_time = mean([r['mape_time_percent'] for r in table])
    avg_tflops = mean([r['mape_tflops_percent'] for r in table])
    avg_ai = mean([r['mape_ai_percent'] for r in table])
    print(f"Average - Time: {avg_time:.2f}% | TFLOPS: {avg_tflops:.2f}% | AI: {avg_ai:.2f}%")


def evaluate(results_dir: str) -> Dict[str, object]:
    samples = load_project1_part1_results(results_dir)
    params = _fit_kernel_scalers(samples)
    traffic_fit = _fit_traffic_correction(samples)

    shortcut_mode_rows: List[Dict[str, float]] = []
    primary_mode_rows: List[Dict[str, float]] = []

    for s in sorted(samples, key=lambda x: (x.cfg, KERNELS.index(x.kernel))):
        pred_shortcut = predict_sample(s, params, use_measured_dram=True)
        pred_primary = predict_sample(s, params, traffic_fit=traffic_fit, use_measured_dram=False)

        # Calculate measured values from actual data
        measured_tflops = s.gflops / 1000.0  # gflops to tflops
        measured_ai = s.flops / s.dram_bytes

        shortcut_mode_rows.append(
            {
                "cfg": s.cfg,
                "kernel": s.kernel,
                "measured_ms": s.time_ms,
                "predicted_ms": pred_shortcut["predicted_time_ms"],
                "tflops": pred_shortcut["tflops"],
                "mape_time_percent": mape(pred_shortcut["predicted_time_ms"], s.time_ms),
                "mape_tflops_percent": mape(pred_shortcut["tflops"], measured_tflops),
                "mape_ai_percent": mape(pred_shortcut["ai_dram"], measured_ai),
                "dram_GB_used": pred_shortcut["dram_bytes_GB"],
                "ai_dram": pred_shortcut["ai_dram"],
                "measured_tflops": measured_tflops,
                "measured_ai": measured_ai,
                "ideal_compute_ms": pred_shortcut["ideal_compute_ms"],
                "ideal_mem_ms": pred_shortcut["ideal_mem_ms"],
            }
        )

        primary_mode_rows.append(
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
                "ai_dram": pred_primary["ai_dram"],
                "measured_tflops": measured_tflops,
                "measured_ai": measured_ai,
                "ideal_compute_ms": pred_primary["ideal_compute_ms"],
                "ideal_mem_ms": pred_primary["ideal_mem_ms"],
            }
        )

    return {
        "samples": samples,
        "kernel_params": params,
        "traffic_fit": traffic_fit,
        "shortcut_mode_rows": shortcut_mode_rows,
        "primary_mode_rows": primary_mode_rows,
    }


def predict_conv_runtime(
    kernel: str,
    B: int,
    Ny: int,
    Nx: int,
    Ni: int,
    Nn: int,
    ky: int = 3,
    kx: int = 3,
    groups: int = 1,
    use_measured_dram: bool = False,
    measured_dram_read_bytes: Optional[float] = None,
    measured_dram_write_bytes: float = 0.0,
    results_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Predict runtime for arbitrary conv layer parameters.

    Notes:
    - Kernel scaling factors are fitted from project1 data (3x3 dense conv).
    - For different structures (e.g., grouped/depthwise, larger kernels), the
      analytic DRAM path is used unless measured bytes are provided.
    """
    if kernel not in KERNELS:
        raise ValueError(f"Unknown kernel '{kernel}'. Expected one of {KERNELS}")
    if groups < 1 or Ni % groups != 0 or Nn % groups != 0:
        raise ValueError("groups must be >=1 and divide both Ni and Nn")

    if results_dir is None:
        results_dir = default_results_dir()
    fit = evaluate(results_dir)
    kernel_params = fit["kernel_params"]
    traffic_fit = fit["traffic_fit"]

    sample = Sample(
        cfg="custom",
        kernel=kernel,
        B=B,
        Ny=Ny,
        Nx=Nx,
        Ni=Ni,
        Nn=Nn,
        ky=ky,
        kx=kx,
        groups=groups,
        time_ms=0.0,
        gflops=0.0,
        dram_read_bytes=float(measured_dram_read_bytes or 0.0),
        dram_write_bytes=float(measured_dram_write_bytes),
    )

    if use_measured_dram and measured_dram_read_bytes is None:
        raise ValueError("measured_dram_read_bytes is required when use_measured_dram=True")

    return predict_sample(
        sample,
        kernel_params,
        traffic_fit=traffic_fit,
        use_measured_dram=use_measured_dram,
    )


def default_results_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "project1", "part1", "results"))


if __name__ == "__main__":
    results_dir = default_results_dir()
    out = evaluate(results_dir)

    print("Calibrated kernel scale factors:")
    for k in KERNELS:
        p = out["kernel_params"][k]
        print(
            f"  {k:<9} scale_compute={p['scale_compute']:<6.2f} "
            f"scale_mem={p['scale_mem']:<6.2f} fit_MAPE={p['mape']:.2f}%"
        )

    print("\nPrimary model: analytic DRAM bytes with a small fitted correction")
    _print_table("Primary model predictions", out["primary_mode_rows"])
    _print_table("Shortcut baseline using measured DRAM bytes", out["shortcut_mode_rows"])

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    write_predictions_csv(os.path.join(out_dir, "predictions_primary_model.csv"), out["primary_mode_rows"])
    write_predictions_csv(os.path.join(out_dir, "predictions_shortcut_model.csv"), out["shortcut_mode_rows"])
    print(f"\nSaved CSV files to: {out_dir}")
