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


def _effective_fp32_tflops() -> float:
    """Return FP32 peak in TFLOPS using HW fields.

    Prefer an explicit value, but derive from device fields when available so
    updates in hw_params.py propagate automatically.
    """
    if "fp32_tflops" in HW:
        return float(HW["fp32_tflops"])

    required = ("num_sms", "cuda_cores_per_sm", "clock_ghz")
    if all(k in HW for k in required):
        return (
            float(HW["num_sms"])
            * float(HW["cuda_cores_per_sm"])
            * 2.0
            * float(HW["clock_ghz"])
            / 1000.0
        )

    raise KeyError("HW must define either fp32_tflops or (num_sms, cuda_cores_per_sm, clock_ghz)")


def _effective_dram_bw_gb_s() -> float:
    """Return DRAM bandwidth in GB/s using HW fields.

    Prefer an explicit calibrated value, but derive from memory clock and bus
    width when available.
    """
    if "dram_bw_gb_s" in HW:
        return float(HW["dram_bw_gb_s"])

    required = ("memory_clock_mhz", "memory_bus_width_bits")
    if all(k in HW for k in required):
        mem_clk_hz = float(HW["memory_clock_mhz"]) * 1e6
        bus_bytes = float(HW["memory_bus_width_bits"]) / 8.0
        return 2.0 * mem_clk_hz * bus_bytes / 1e9

    raise KeyError("HW must define either dram_bw_gb_s or (memory_clock_mhz, memory_bus_width_bits)")


FP32_TFLOPS = _effective_fp32_tflops()
DRAM_BW_GB_S = _effective_dram_bw_gb_s()


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


def _hierarchical_ideal_times_ms(flops: float, dram_bytes: float, sample: Optional[Sample] = None) -> Tuple[float, float, float, float]:
    """Return ideal times (ms) for compute, L1/SPAD, L2 and DRAM.

    The four levels correspond to:
    - compute: theoretical time to execute all FLOPs at peak FP32 TFLOPS.
    - l1/spad: time to serve data that lives in on-chip scratchpad/L1 cache.
    - l2: time to serve data that is reused from L2 (fits in L2 between accesses).
    - dram: time to transfer bytes that must come from off-chip HBM.

    L2 traffic = analytic_minimum_traffic - actual_dram_traffic when
    analytic < dram (correction > 1 means extra cache-line waste goes to DRAM;
    correction < 1 means some traffic is served from L2, not DRAM).

    For kernels where the correction inflates DRAM above the analytic minimum
    (cache-line waste), any tensor that fits in L2 is still being re-served
    from L2 across tile iterations — we estimate that separately.
    """
    # Compute-bound time
    t_compute_ms = flops / (FP32_TFLOPS * 1e12) * 1e3

    # DRAM time (off-chip bandwidth)
    t_dram_ms = dram_bytes / (DRAM_BW_GB_S * 1e9) * 1e3

    # L2 traffic estimation
    # Case A: correction < 1 → corrected DRAM < analytic minimum → some traffic served from L2
    # Case B: correction > 1 → corrected DRAM > analytic minimum → cache-line waste; tensors
    #         small enough to fit in L2 are still re-served from L2 across tile iterations.
    l2_bw = float(HW.get("l2_bw_gb_s", 1.0)) * 1e9
    l1_bw = float(HW.get("l1_bw_gb_s", 1.0)) * 1e9
    l2_cap = float(HW.get("l2_bytes", 0))

    if sample is not None:
        try:
            analytic_total_bytes = _analytic_dram_bytes(sample)
        except Exception:
            analytic_total_bytes = dram_bytes

        # Case A: analytic < dram (correction reduced traffic → L2 serves the remainder)
        l2_from_reuse = max(0.0, analytic_total_bytes - dram_bytes)

        # Case B: estimate L2 traffic from tensors that fit in L2 cache
        # These tensors are loaded from DRAM once per kernel launch, then served
        # from L2 on subsequent accesses within the same launch.
        syn_b, inp_b, out_b = conv_tensor_bytes(
            sample.B, sample.Ny, sample.Nx, sample.Ni, sample.Nn,
            sample.ky, sample.kx, sample.groups,
        )
        # Weight tensor: if it fits in L2, subsequent blocks read it from L2, not DRAM.
        # Extra L2 traffic ≈ total analytic weight accesses minus the one initial DRAM load.
        l2_from_weight_reuse = 0.0
        if syn_b < l2_cap:
            # Approximate: weight re-reads from L2 = weight_analytic_total - syn_b (initial load)
            total_weight_analytic = analytic_total_bytes - inp_b - out_b
            l2_from_weight_reuse = max(0.0, total_weight_analytic - syn_b)

        l2_bytes = max(l2_from_reuse, l2_from_weight_reuse)
    else:
        l2_bytes = 0.0

    # L1/SPAD: data that stays on-chip in shared memory or L1 within a block.
    # Conservatively cap at total L1+SPAD capacity across all SMs.
    if sample is not None:
        try:
            analytic_total_bytes = _analytic_dram_bytes(sample)
        except Exception:
            analytic_total_bytes = dram_bytes
    else:
        analytic_total_bytes = dram_bytes
    l1_cap_total = float(HW.get("l1_spad_bytes_per_sm", 0)) * float(HW.get("num_sms", 1))
    l1_bytes = min(analytic_total_bytes, l1_cap_total)

    t_l2_ms = l2_bytes / l2_bw * 1e3
    t_l1_ms = l1_bytes / l1_bw * 1e3

    return t_compute_ms, t_l1_ms, t_l2_ms, t_dram_ms


# Backwards-compatible wrapper for earlier call sites that expected two values.
def _ideal_times_ms(flops: float, dram_bytes: float) -> Tuple[float, float]:
    c, _, _, d = _hierarchical_ideal_times_ms(flops, dram_bytes)
    return c, d


def _fit_kernel_scalers(
    rows: List[Sample],
    use_measured_dram: bool,
    traffic_fit: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Fit per-kernel scale factors for compute and DRAM ideal times.

    Model form:
      pred_ms = max(scale_compute * ideal_compute_ms,
                    scale_mem * ideal_mem_ms)

    The fitting path should match the prediction path:
    - shortcut baseline: measured DRAM bytes
    - primary model: corrected analytic DRAM bytes
    """
    if not use_measured_dram and traffic_fit is None:
        raise ValueError("traffic_fit is required when fitting with analytic DRAM")

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
                    if use_measured_dram:
                        dram_bytes = r.dram_bytes
                    else:
                        dram_bytes = _corrected_analytic_dram_bytes(r, traffic_fit)
                    c_ms, l1_ms, l2_ms, m_ms = _hierarchical_ideal_times_ms(r.flops, dram_bytes, sample=r)
                    # True hierarchical bottleneck: compute, L2, and DRAM compete.
                    # L2 is unscaled (scale_l2 = 1.0): assume L2 achieves rated BW.
                    pred = max(sc * c_ms, l2_ms, sm * m_ms)
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

    c_ms, l1_ms, l2_ms, m_ms = _hierarchical_ideal_times_ms(sample.flops, dram_bytes, sample=sample)

    p = kernel_params[sample.kernel]
    sc_t = p["scale_compute"] * c_ms
    sm_t = p["scale_mem"] * m_ms
    # True hierarchical bottleneck: L2 is included at rated bandwidth (scale_l2 = 1.0).
    pred_ms = max(sc_t, l2_ms, sm_t)

    # Identify which level is the bottleneck
    if pred_ms == sc_t:
        bottleneck = "compute"
    elif pred_ms == l2_ms:
        bottleneck = "l2"
    else:
        bottleneck = "memory"

    tflops = sample.flops / (pred_ms * 1e9)
    achieved_ai = sample.flops / dram_bytes

    return {
        "predicted_time_ms": pred_ms,
        "tflops": tflops,
        "tops": tflops,
        "achieved_ai": achieved_ai,
        "ai_dram": sample.flops / dram_bytes,
        "ideal_compute_ms": c_ms,
        "ideal_l1_ms": l1_ms,
        "ideal_l2_ms": l2_ms,
        "ideal_mem_ms": m_ms,
        "dram_bytes_GB": dram_bytes / 1e9,
        "bottleneck": bottleneck,
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
        "tops",
        "mape_time_percent",
        "mape_tflops_percent",
        "mape_ai_percent",
        "dram_GB_used",
        "measured_tflops",
        "measured_ai",
        "achieved_ai",
        "ai_dram",
        "ideal_compute_ms",
        "ideal_l1_ms",
        "ideal_l2_ms",
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


def _print_achieved_metrics(title: str, table: List[Dict[str, float]]) -> None:
    print("\n" + title)
    print("-" * len(title))
    print(f"{'cfg':<6} {'kernel':<9} {'TOps':>10} {'AI(DRAM)':>12} {'Pred ms':>12}")
    for r in table:
        print(
            f"{r['cfg']:<6} {r['kernel']:<9} {r['tflops']:>10.3f} "
            f"{r['ai_dram']:>12.3f} {r['predicted_ms']:>12.3f}"
        )


def evaluate(results_dir: str) -> Dict[str, object]:
    samples = load_project1_part1_results(results_dir)
    traffic_fit = _fit_traffic_correction(samples)
    params_shortcut = _fit_kernel_scalers(samples, use_measured_dram=True)
    params_primary = _fit_kernel_scalers(samples, use_measured_dram=False, traffic_fit=traffic_fit)

    shortcut_mode_rows: List[Dict[str, float]] = []
    primary_mode_rows: List[Dict[str, float]] = []

    for s in sorted(samples, key=lambda x: (x.cfg, KERNELS.index(x.kernel))):
        pred_shortcut = predict_sample(s, params_shortcut, use_measured_dram=True)
        pred_primary = predict_sample(s, params_primary, traffic_fit=traffic_fit, use_measured_dram=False)

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
                "tops": pred_shortcut["tops"],
                "mape_time_percent": mape(pred_shortcut["predicted_time_ms"], s.time_ms),
                "mape_tflops_percent": mape(pred_shortcut["tflops"], measured_tflops),
                "mape_ai_percent": mape(pred_shortcut["ai_dram"], measured_ai),
                "dram_GB_used": pred_shortcut["dram_bytes_GB"],
                "achieved_ai": pred_shortcut["achieved_ai"],
                "ai_dram": pred_shortcut["ai_dram"],
                "measured_tflops": measured_tflops,
                "measured_ai": measured_ai,
                "ideal_compute_ms": pred_shortcut["ideal_compute_ms"],
                "ideal_l1_ms": pred_shortcut.get("ideal_l1_ms", 0.0),
                "ideal_l2_ms": pred_shortcut.get("ideal_l2_ms", 0.0),
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
                "tops": pred_primary["tops"],
                "mape_time_percent": mape(pred_primary["predicted_time_ms"], s.time_ms),
                "mape_tflops_percent": mape(pred_primary["tflops"], measured_tflops),
                "mape_ai_percent": mape(pred_primary["ai_dram"], measured_ai),
                "dram_GB_used": pred_primary["dram_bytes_GB"],
                "achieved_ai": pred_primary["achieved_ai"],
                "ai_dram": pred_primary["ai_dram"],
                "measured_tflops": measured_tflops,
                "measured_ai": measured_ai,
                "ideal_compute_ms": pred_primary["ideal_compute_ms"],
                "ideal_l1_ms": pred_primary.get("ideal_l1_ms", 0.0),
                "ideal_l2_ms": pred_primary.get("ideal_l2_ms", 0.0),
                "ideal_mem_ms": pred_primary["ideal_mem_ms"],
            }
        )

    return {
        "samples": samples,
        # Backward-compatible alias now points to primary-mode params.
        "kernel_params": params_primary,
        "kernel_params_shortcut": params_shortcut,
        "kernel_params_primary": params_primary,
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
    traffic_fit = fit["traffic_fit"]
    kernel_params = fit["kernel_params_shortcut"] if use_measured_dram else fit["kernel_params_primary"]

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


# ── Occupancy and coalescing models for K3 smem_wi ───────────────────────────

def compute_smwi_occupancy(
    smwi_tile_ni: int,
    ky: int = 3,
    kx: int = 3,
    tile_x: int = 16,
    tile_y: int = 16,
) -> dict:
    """Compute SM occupancy for K3 smem_wi as a function of smwi_tile_ni.

    Shared memory per block = smwi_tile_ni * (ky*kx + inp_tile_y*inp_tile_x) * 4 bytes.
    Occupancy is limited by whichever resource runs out first: shared memory or threads.
    """
    inp_tile_y = tile_y + ky - 1
    inp_tile_x = tile_x + kx - 1
    smem_per_block = smwi_tile_ni * (ky * kx + inp_tile_y * inp_tile_x) * 4

    l1_bytes = int(HW["l1_spad_bytes_per_sm"])          # 96 KB unified
    max_blocks_smem = l1_bytes // smem_per_block
    max_blocks_threads = int(HW["max_threads_per_sm"]) // (tile_x * tile_y)  # 8
    max_blocks_hw = int(HW["max_blocks_per_sm"])          # 32

    blocks_per_sm = min(max_blocks_smem, max_blocks_threads, max_blocks_hw)

    return {
        "smem_bytes": smem_per_block,
        "max_blocks_smem": max_blocks_smem,
        "max_blocks_threads": max_blocks_threads,
        "blocks_per_sm": max(blocks_per_sm, 1),
        "occupancy": max(blocks_per_sm, 1) / max_blocks_threads,
        "bottleneck": "smem" if max_blocks_smem < max_blocks_threads else "threads",
    }


def smwi_coalesce_efficiency(smwi_tile_ni: int) -> float:
    """Cache-line utilization for K3 smem_wi input loads.

    A 32-thread warp splits into (32 / smwi_tile_ni) sub-groups, each loading
    smwi_tile_ni consecutive floats from the same (ly, lx) position.
    A 128-byte cache line holds 32 floats; each sub-group uses
    min(smwi_tile_ni, 32) / 32 of the line.

    A dampened exponent (0.6) is applied by the caller to account for partial
    L1-cache reuse: nearby warps often share cache lines, partially recovering
    the wasted bandwidth.
    """
    return min(smwi_tile_ni, 32) / 32.0


def predict_smwi_tile_sweep(
    B: int,
    Ny: int,
    Nx: int,
    Ni: int,
    Nn: int,
    tile_ni_list: list,
    results_dir: Optional[str] = None,
    ref_tile_ni: int = 8,
    coalesce_exp: float = 0.6,
) -> list:
    """Predict K3 smem_wi runtime for a sweep of smwi_tile_ni values.

    Mechanistic corrections applied on top of the fitted base model:
      1. Coalescing: fewer bytes per cache line → (ref_eff/cur_eff)^coalesce_exp × t_mem
      2. Occupancy:  fewer blocks/SM  → sqrt(ref_occ/cur_occ) × t_compute
         (sqrt dampening: memory-BW-saturated kernels hide occupancy effects)
    """
    if results_dir is None:
        results_dir = default_results_dir()

    fit = evaluate(results_dir)
    params = fit["kernel_params_primary"]["smem_wi"]
    traffic_fit = fit["traffic_fit"]
    scale_compute = params["scale_compute"]
    scale_mem = params["scale_mem"]

    # Reference sample at ref_tile_ni (used to get base ideal times)
    ref_sample = Sample(
        cfg="ref", kernel="smem_wi",
        B=B, Ny=Ny, Nx=Nx, Ni=Ni, Nn=Nn,
        ky=3, kx=3, groups=1,
        time_ms=0.0, gflops=0.0,
        dram_read_bytes=0.0, dram_write_bytes=0.0,
    )
    ref_dram = _corrected_analytic_dram_bytes(ref_sample, traffic_fit)
    flops = conv_flops(B, Ny, Nx, Ni, Nn, 3, 3)

    t_compute_ideal = flops / (FP32_TFLOPS * 1e12) * 1e3   # ms, constant
    t_mem_ideal_ref = ref_dram / (DRAM_BW_GB_S * 1e9) * 1e3  # ms at ref tni

    # Reference occupancy and coalescing
    ref_occ_info = compute_smwi_occupancy(ref_tile_ni)
    ref_occ = ref_occ_info["occupancy"]
    ref_coalesce = smwi_coalesce_efficiency(ref_tile_ni)

    results = []
    for tni in tile_ni_list:
        occ_info = compute_smwi_occupancy(tni)
        cur_occ = occ_info["occupancy"]
        cur_coalesce = smwi_coalesce_efficiency(tni)

        # Memory term: coalescing correction (dampened by coalesce_exp)
        coalesce_factor = (ref_coalesce / cur_coalesce) ** coalesce_exp
        t_mem_new = t_mem_ideal_ref * coalesce_factor

        # Compute term: occupancy correction (sqrt dampening for BW-bound kernels)
        occ_factor = math.sqrt(ref_occ / cur_occ)
        t_compute_new = t_compute_ideal * occ_factor

        predicted_ms = max(scale_compute * t_compute_new, scale_mem * t_mem_new)

        results.append({
            "tile_ni": tni,
            "predicted_ms": predicted_ms,
            "occupancy": cur_occ,
            "blocks_per_sm": occ_info["blocks_per_sm"],
            "coalesce_eff": cur_coalesce,
            "coalesce_factor": coalesce_factor,
            "occ_factor": occ_factor,
            "bottleneck": "compute" if scale_compute * t_compute_new >= scale_mem * t_mem_new else "memory",
        })

    return results


def predict_smwi_problem_sweep(
    Ny_list: list,
    B: int = 16,
    Ni: int = 64,
    Nn: int = 64,
    results_dir: Optional[str] = None,
) -> list:
    """Predict K3 smem_wi runtime for varying spatial problem sizes (Ny = Nx).

    Uses the primary analytic model (no measured DRAM needed), so it generalises
    to arbitrary spatial dimensions not seen during fitting.
    """
    if results_dir is None:
        results_dir = default_results_dir()

    fit = evaluate(results_dir)
    params = fit["kernel_params_primary"]
    traffic_fit = fit["traffic_fit"]

    results = []
    for Ny in Ny_list:
        Nx = Ny
        sample = Sample(
            cfg="sweep", kernel="smem_wi",
            B=B, Ny=Ny, Nx=Nx, Ni=Ni, Nn=Nn,
            ky=3, kx=3, groups=1,
            time_ms=0.0, gflops=0.0,
            dram_read_bytes=0.0, dram_write_bytes=0.0,
        )
        pred = predict_sample(sample, params, traffic_fit=traffic_fit, use_measured_dram=False)
        results.append({
            "Ny": Ny,
            "predicted_ms": pred["predicted_time_ms"],
            "tflops": pred["tflops"],
            "ai_dram": pred["ai_dram"],
            "bottleneck": pred.get("bottleneck", "unknown"),
        })

    return results


if __name__ == "__main__":
    results_dir = default_results_dir()
    out = evaluate(results_dir)

    print("Calibrated kernel scale factors (primary analytic path):")
    for k in KERNELS:
        p = out["kernel_params_primary"][k]
        print(
            f"  {k:<9} scale_compute={p['scale_compute']:<6.2f} "
            f"scale_mem={p['scale_mem']:<6.2f} fit_MAPE={p['mape']:.2f}%"
        )

    print("\nCalibrated kernel scale factors (shortcut measured-DRAM path):")
    for k in KERNELS:
        p = out["kernel_params_shortcut"][k]
        print(
            f"  {k:<9} scale_compute={p['scale_compute']:<6.2f} "
            f"scale_mem={p['scale_mem']:<6.2f} fit_MAPE={p['mape']:.2f}%"
        )

    print("\nPrimary model: analytic DRAM bytes with a small fitted correction")
    _print_table("Primary model predictions", out["primary_mode_rows"])
    _print_table("Shortcut baseline using measured DRAM bytes", out["shortcut_mode_rows"])
    _print_achieved_metrics("Primary model achieved throughput and intensity", out["primary_mode_rows"])

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    write_predictions_csv(os.path.join(out_dir, "predictions_primary_model.csv"), out["primary_mode_rows"])
    write_predictions_csv(os.path.join(out_dir, "predictions_shortcut_model.csv"), out["shortcut_mode_rows"])
    print(f"\nSaved CSV files to: {out_dir}")
