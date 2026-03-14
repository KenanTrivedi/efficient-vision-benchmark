#!/usr/bin/env python3
"""
02_generate_visualizations.py - benchmark plots and static site data.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.models import load_model_config


RESULTS_FILE = Path("results/benchmark_results.json")
FINETUNE_RESULTS_FILE = Path("results/finetune/summary.json")
FIGURES_DIR = Path("results/figures")
DOCS_DIR = Path("docs")
DOCS_ASSETS_DIR = DOCS_DIR / "assets"
DOCS_FIGURES_DIR = DOCS_ASSETS_DIR / "figures"
DOCS_SITE_DATA_FILE = DOCS_DIR / "site_data.json"
DOCS_SITE_SCRIPT_FILE = DOCS_DIR / "site_data.js"

BACKGROUND = "#f4efe5"
PANEL = "#fffaf2"
INK = "#18212b"
SUBTLE = "#6b7280"
GRID = "#d8d2c6"

PALETTE = [
    "#0f766e",
    "#b45309",
    "#1d4ed8",
    "#dc2626",
    "#6d28d9",
    "#15803d",
    "#7c3aed",
    "#be123c",
    "#334155",
]

MODEL_CONFIG = load_model_config()


def load_json(path: Path) -> Dict:
    if not path.exists():
        print(f"\n  ERROR: File not found: {path}")
        print("    Run python 01_run_benchmark.py first.\n")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_finetune_summary() -> Dict | None:
    if FINETUNE_RESULTS_FILE.exists():
        with open(FINETUNE_RESULTS_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return None


def extract_meta(payload: Dict) -> Dict:
    meta = dict(payload.get("meta", {}))
    if "timestamp" not in meta and "timestamp" in payload:
        meta["timestamp"] = payload.get("timestamp")
    if "device" not in meta and "device" in payload:
        meta["device"] = payload.get("device")
    meta.setdefault("dataset", "EuroSAT")
    meta.setdefault("protocol", "head_reset_transfer_baseline")
    meta.setdefault("protocol_label", "Head-reset transfer baseline")
    meta.setdefault(
        "protocol_description",
        "Each pretrained backbone is adapted to EuroSAT by replacing the final classifier with a fresh 10-class layer and evaluating before training.",
    )
    return meta


def normalize_model_records(payload: Dict, experiment_name: str = "baseline") -> List[Dict]:
    records = []
    meta = extract_meta(payload)
    protocol_label = meta.get("protocol_label", "Benchmark run")

    for model_key, model_payload in payload.get("models", {}).items():
        model_cfg = MODEL_CONFIG.get(model_key, {})
        model_info = model_payload.get("model_info", {})
        flops = model_payload.get("flops", {})
        throughput = model_payload.get("throughput", {})
        accuracy = model_payload.get("accuracy", {})
        latency_cpu = model_payload.get("latency_cpu", {})
        latency_gpu = model_payload.get("latency_gpu") or {}

        params_m = round(model_info.get("total_params", 0) / 1e6, 2)
        top1 = accuracy.get("top1_accuracy")
        cpu_latency = latency_cpu.get("median_ms")

        if top1 is None or cpu_latency is None:
            continue

        records.append({
            "experiment": experiment_name,
            "experiment_label": protocol_label,
            "model_key": model_key,
            "name": model_payload.get("name") or model_cfg.get("name", model_key),
            "year": model_payload.get("year") or model_cfg.get("year"),
            "source": model_payload.get("source") or model_cfg.get("source"),
            "top1_accuracy": round(float(top1), 2),
            "top5_accuracy": accuracy.get("top5_accuracy"),
            "cpu_latency_ms": round(float(cpu_latency), 2),
            "gpu_latency_ms": latency_gpu.get("median_ms"),
            "params_m": params_m,
            "size_mb": model_info.get("size_mb"),
            "macs_m": round((flops.get("macs") or 0) / 1e6, 2) if flops.get("macs") is not None else None,
            "throughput_bs1": throughput.get("1"),
            "throughput_bs16": throughput.get("16"),
            "efficiency_acc_per_mparam": round(float(top1) / max(params_m, 0.1), 3),
            "efficiency_acc_per_ms": round(float(top1) / max(float(cpu_latency), 0.1), 3),
        })

    records.sort(key=lambda record: (record["cpu_latency_ms"], -record["top1_accuracy"]))
    return records


def compute_pareto_front(records: List[Dict]) -> List[Dict]:
    frontier = []
    best_accuracy = -math.inf

    for record in sorted(records, key=lambda item: item["cpu_latency_ms"]):
        if record["top1_accuracy"] > best_accuracy:
            frontier.append(record)
            best_accuracy = record["top1_accuracy"]

    return frontier


def bubble_size(params_m: float) -> float:
    return 140 + 55 * math.sqrt(max(params_m, 0.1))


def label_offsets(index: int) -> tuple[int, int]:
    x_offsets = [10, 12, -12, 14, -14, 16, -16, 18, -18]
    y_offsets = [12, -14, 18, -18, 10, -10, 16, -16, 22]
    return x_offsets[index % len(x_offsets)], y_offsets[index % len(y_offsets)]


def _apply_figure_style(fig, ax) -> None:
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(PANEL)
    ax.grid(True, color=GRID, alpha=0.85, linewidth=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    ax.tick_params(colors=INK)


def plot_accuracy_vs_latency(records: List[Dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.6, 7.2))
    _apply_figure_style(fig, ax)

    frontier = compute_pareto_front(records)
    frontier_x = [record["cpu_latency_ms"] for record in frontier]
    frontier_y = [record["top1_accuracy"] for record in frontier]
    ax.plot(frontier_x, frontier_y, color="#1f2937", linewidth=1.8, linestyle="--", alpha=0.75, label="Pareto front")

    for index, record in enumerate(records):
        x_value = record["cpu_latency_ms"]
        y_value = record["top1_accuracy"]
        point_size = bubble_size(record["params_m"])
        color = PALETTE[index % len(PALETTE)]

        ax.scatter(
            x_value,
            y_value,
            s=point_size,
            color=color,
            alpha=0.88,
            edgecolors="white",
            linewidths=1.2,
            zorder=3,
        )

        offset_x, offset_y = label_offsets(index)
        ax.annotate(
            record["name"],
            (x_value, y_value),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=9,
            color=INK,
            bbox={
                "boxstyle": "round,pad=0.25",
                "fc": "#fffdf8",
                "ec": "#d6d3d1",
                "alpha": 0.92,
            },
        )

    ax.set_xscale("log")
    ax.set_xlabel("CPU inference latency (ms, log scale)")
    ax.set_ylabel("Top-1 accuracy on EuroSAT test split (%)")
    ax.set_title("EuroSAT Transfer Baseline: Accuracy vs. CPU Latency", fontsize=15, color=INK, pad=12)
    ax.text(
        0.01,
        0.98,
        "Bubble area encodes parameter count.\nDashed line marks the non-dominated latency/accuracy frontier.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=SUBTLE,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_footprint(records: List[Dict], output_path: Path) -> None:
    sorted_records = sorted(records, key=lambda record: record["params_m"], reverse=True)
    names = [record["name"] for record in sorted_records]
    params = [record["params_m"] for record in sorted_records]
    colors = [PALETTE[index % len(PALETTE)] for index in range(len(sorted_records))]

    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    _apply_figure_style(fig, ax)

    bars = ax.barh(names, params, color=colors, edgecolor="#ffffff", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (millions, log scale)")
    ax.set_title("Model Footprint Spread", fontsize=15, color=INK, pad=12)

    for bar, value in zip(bars, params):
        ax.text(
            value * 1.04,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}M",
            va="center",
            fontsize=9,
            color=INK,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_latency_breakdown(records: List[Dict], output_path: Path) -> None:
    gpu_records = [record for record in records if record["gpu_latency_ms"] is not None]
    if not gpu_records:
        return

    ordered = sorted(gpu_records, key=lambda record: record["cpu_latency_ms"])
    y_positions = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    _apply_figure_style(fig, ax)

    for index, record in enumerate(ordered):
        cpu_latency = record["cpu_latency_ms"]
        gpu_latency = record["gpu_latency_ms"]
        color = PALETTE[index % len(PALETTE)]

        ax.plot([gpu_latency, cpu_latency], [index, index], color="#cbd5e1", linewidth=2.2, zorder=1)
        ax.scatter(gpu_latency, index, color=color, s=85, zorder=3, edgecolors="white", linewidths=0.8)
        ax.scatter(cpu_latency, index, color="#111827", s=55, zorder=3)

    ax.set_yticks(y_positions, [record["name"] for record in ordered])
    ax.set_xscale("log")
    ax.set_xlabel("Median latency (ms, log scale)")
    ax.set_title("CPU vs. GPU Inference Latency", fontsize=15, color=INK, pad=12)
    ax.text(
        0.01,
        0.98,
        "Colored marker = GPU, black marker = CPU",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=SUBTLE,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_finetune_accuracy_delta(baseline_records: List[Dict], finetune_summary: Dict, output_path: Path) -> bool:
    if not finetune_summary:
        return False

    finetuned_models = finetune_summary.get("models", {})
    comparison_rows = []

    for record in baseline_records:
        finetuned = finetuned_models.get(record["model_key"])
        if not finetuned:
            continue
        comparison_rows.append({
            "name": record["name"],
            "baseline_accuracy": record["top1_accuracy"],
            "finetuned_accuracy": finetuned.get("test_metrics", {}).get("top1_accuracy"),
        })

    comparison_rows = [
        row for row in comparison_rows
        if row["finetuned_accuracy"] is not None
    ]
    if not comparison_rows:
        return False

    names = [row["name"] for row in comparison_rows]
    baseline_values = [row["baseline_accuracy"] for row in comparison_rows]
    finetuned_values = [row["finetuned_accuracy"] for row in comparison_rows]
    x_positions = np.arange(len(names))
    width = 0.34

    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    _apply_figure_style(fig, ax)
    ax.bar(x_positions - width / 2, baseline_values, width=width, color="#cbd5e1", label="Baseline")
    ax.bar(x_positions + width / 2, finetuned_values, width=width, color="#0f766e", label="Fine-tuned")

    ax.set_xticks(x_positions, names, rotation=12, ha="right")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Accuracy Lift After Fine-Tuning", fontsize=15, color=INK, pad=12)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return True


def summarize_records(records: List[Dict]) -> Dict:
    if not records:
        return {}

    best_accuracy = max(records, key=lambda item: item["top1_accuracy"])
    fastest_cpu = min(records, key=lambda item: item["cpu_latency_ms"])
    best_size_efficiency = max(records, key=lambda item: item["efficiency_acc_per_mparam"])

    return {
        "model_count": len(records),
        "best_accuracy_model": {
            "name": best_accuracy["name"],
            "value": best_accuracy["top1_accuracy"],
            "unit": "%",
        },
        "fastest_cpu_model": {
            "name": fastest_cpu["name"],
            "value": fastest_cpu["cpu_latency_ms"],
            "unit": "ms",
        },
        "best_size_efficiency_model": {
            "name": best_size_efficiency["name"],
            "value": best_size_efficiency["efficiency_acc_per_mparam"],
            "unit": "acc / M param",
        },
        "recommended_finetune_targets": [
            "mobilevitv2_050",
            "mobilenet_v3_large",
            "convnext_tiny",
        ],
    }


def copy_generated_assets(figure_paths: List[Path]) -> None:
    DOCS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for source_path in figure_paths:
        if not source_path.exists():
            continue
        destination_path = DOCS_FIGURES_DIR / source_path.name
        shutil.copy2(source_path, destination_path)


def write_site_data(
    baseline_payload: Dict,
    baseline_records: List[Dict],
    figure_paths: List[Path],
    finetune_summary: Dict | None,
) -> None:
    summary = summarize_records(baseline_records)
    meta = extract_meta(baseline_payload)

    comparison = {}
    if finetune_summary:
        comparison["available"] = True
        comparison["completed_models"] = sorted(list(finetune_summary.get("models", {}).keys()))
    else:
        comparison["available"] = False
        comparison["completed_models"] = []

    site_payload = {
        "generated_at": meta.get("timestamp"),
        "meta": meta,
        "summary": summary,
        "models": baseline_records,
        "figures": {
            "accuracy_vs_latency": f"assets/figures/{figure_paths[0].name}",
            "parameter_footprint": f"assets/figures/{figure_paths[1].name}",
            "latency_breakdown": f"assets/figures/{figure_paths[2].name}",
        },
        "finetune": comparison,
        "job_alignment": {
            "role_url": "https://career.quantum-systems.com/o/ai-software-engineer-mfd",
            "highlights": [
                "Preparing and selecting data, training and validating models, and deploying them on embedded UxV platforms.",
                "Experience with quantization, pruning, and distillation for resource-constrained environments.",
                "Hands-on deployment on NVIDIA Jetson, ARM, FPGA, or custom SoC edge hardware.",
            ],
        },
    }

    if finetune_summary and "accuracy_comparison" in finetune_summary:
        site_payload["finetune"]["accuracy_comparison"] = finetune_summary["accuracy_comparison"]

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DOCS_SITE_DATA_FILE, "w", encoding="utf-8") as handle:
        json.dump(site_payload, handle, indent=2)
    with open(DOCS_SITE_SCRIPT_FILE, "w", encoding="utf-8") as handle:
        handle.write("window.BENCHMARK_SITE_DATA = ")
        json.dump(site_payload, handle, indent=2)
        handle.write(";\n")


def main() -> None:
    print()
    print("  ==========================================================")
    print("    VISUALIZATION + DASHBOARD GENERATOR")
    print("    Building plots and GitHub Pages data assets")
    print("  ==========================================================")
    print()

    benchmark_payload = load_json(RESULTS_FILE)
    finetune_summary = load_finetune_summary()
    records = normalize_model_records(benchmark_payload, experiment_name="baseline")

    if not records:
        print("  ERROR: No complete model records were found in results/benchmark_results.json")
        sys.exit(1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    accuracy_vs_latency = FIGURES_DIR / "accuracy_vs_latency.png"
    parameter_footprint = FIGURES_DIR / "parameter_footprint.png"
    latency_breakdown = FIGURES_DIR / "cpu_gpu_latency_breakdown.png"
    finetune_delta = FIGURES_DIR / "baseline_vs_finetuned_accuracy.png"
    stale_radar = FIGURES_DIR / "tradeoff_radar.png"

    benchmark_meta = extract_meta(benchmark_payload)
    print(f"  Dataset   : {benchmark_meta.get('dataset', 'N/A')}")
    print(f"  Protocol  : {benchmark_meta.get('protocol_label', 'N/A')}")
    print(f"  Models    : {len(records)}")
    print()

    print("  [1]  Accuracy vs. latency scatter ...          ", end="", flush=True)
    plot_accuracy_vs_latency(records, accuracy_vs_latency)
    print(f"OK  {accuracy_vs_latency}")

    print("  [2]  Parameter footprint plot ...             ", end="", flush=True)
    plot_parameter_footprint(records, parameter_footprint)
    print(f"OK  {parameter_footprint}")

    print("  [3]  CPU/GPU latency comparison ...           ", end="", flush=True)
    plot_latency_breakdown(records, latency_breakdown)
    print(f"OK  {latency_breakdown}")

    generated_figures = [accuracy_vs_latency, parameter_footprint, latency_breakdown]

    if plot_finetune_accuracy_delta(records, finetune_summary, finetune_delta):
        generated_figures.append(finetune_delta)
        print(f"  [4]  Fine-tune accuracy comparison ...        OK  {finetune_delta}")

    if stale_radar.exists():
        stale_radar.unlink()
        print("  [x]  Removed stale radar chart from previous CIFAR-era output.")

    copy_generated_assets(generated_figures)
    write_site_data(benchmark_payload, records, generated_figures[:3], finetune_summary)

    print()
    print("  ------------------------------------------------")
    print("  Figures written to results/figures/")
    print("  Dashboard data written to docs/site_data.json and docs/site_data.js")
    print("  ------------------------------------------------")
    print()


if __name__ == "__main__":
    main()
