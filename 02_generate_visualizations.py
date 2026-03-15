#!/usr/bin/env python3
"""
02_generate_visualizations.py - Benchmark plots and static site data.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import DEFAULT_SEED, TransformedSubset, _load_base_dataset, get_or_create_eurosat_splits, get_transforms
from src.models import load_model_config
from src.models import get_model
from src.training import seed_everything


RESULTS_FILE = Path("results/benchmark_results.json")
FINETUNE_RESULTS_FILE = Path("results/finetune/summary.json")
DEPLOYMENT_RESULTS_FILE = Path("results/deployment/summary.json")
FIGURES_DIR = Path("results/figures")
DOCS_DIR = Path("docs")
DOCS_ASSETS_DIR = DOCS_DIR / "assets"
DOCS_FIGURES_DIR = DOCS_ASSETS_DIR / "figures"
DOCS_SITE_DATA_FILE = DOCS_DIR / "site_data.json"
DOCS_SITE_SCRIPT_FILE = DOCS_DIR / "site_data.js"
PDF_OUTPUT_FILE = DOCS_ASSETS_DIR / "EuroSAT_Edge_Vision_Benchmark_Portfolio.pdf"

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
QUALITATIVE_SAMPLE_COUNT = 6
QUALITATIVE_PREFERRED_CLASSES = [
    "AnnualCrop",
    "Highway",
    "Industrial",
    "Residential",
    "River",
    "SeaLake",
]


def load_json(path: Path) -> Dict:
    if not path.exists():
        print(f"\n  ERROR: File not found: {path}")
        print("    Run python 01_run_benchmark.py first.\n")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_optional_json(path: Path) -> Dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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
        "Each pretrained backbone is adapted to EuroSAT by replacing the final classifier with a fresh task-specific head and evaluating before training.",
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

    comparison_rows = [row for row in comparison_rows if row["finetuned_accuracy"] is not None]
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


def prettify_class_name(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name)


def load_model_for_inference(
    model_key: str,
    *,
    num_classes: int,
    device: str,
    checkpoint_path: str | None = None,
) -> torch.nn.Module:
    model = get_model(model_key, num_classes=num_classes, pretrained=True)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def choose_qualitative_model_key(
    finetune_summary: Dict | None,
    deployment_summary: Dict | None,
) -> str | None:
    selected_model = (deployment_summary or {}).get("selected_model") or {}
    if selected_model.get("model_key"):
        return selected_model["model_key"]

    leaderboard = (finetune_summary or {}).get("leaderboard") or []
    if leaderboard:
        return leaderboard[0]["model_key"]

    return None


def build_qualitative_examples(
    baseline_payload: Dict,
    finetune_summary: Dict | None,
    deployment_summary: Dict | None,
    *,
    data_dir: str = "./data",
) -> Dict | None:
    model_key = choose_qualitative_model_key(finetune_summary, deployment_summary)
    if not model_key or not finetune_summary:
        return None

    finetuned_model_summary = (finetune_summary.get("models") or {}).get(model_key)
    if not finetuned_model_summary:
        return None

    checkpoint_path = ((finetuned_model_summary.get("artifacts") or {}).get("checkpoint"))
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return None

    split_payload = get_or_create_eurosat_splits(data_dir=data_dir, seed=DEFAULT_SEED)
    base_dataset = _load_base_dataset(data_dir=data_dir)
    class_names = split_payload["class_names"]
    test_indices = list(split_payload["indices"]["test"])
    num_classes = len(class_names)

    model_name = finetuned_model_summary.get("name") or baseline_payload.get("models", {}).get(model_key, {}).get("name", model_key)
    input_size = int(
        finetuned_model_summary.get("input_size")
        or baseline_payload.get("models", {}).get(model_key, {}).get("input_size")
        or 224
    )
    eval_transform = get_transforms(input_size=input_size)[1]
    eval_dataset = TransformedSubset(base_dataset, test_indices, transform=eval_transform)
    dataloader = DataLoader(
        eval_dataset,
        batch_size=96,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(DEFAULT_SEED)
    baseline_model = load_model_for_inference(model_key, num_classes=num_classes, device=device)
    finetuned_model = load_model_for_inference(
        model_key,
        num_classes=num_classes,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    labels_all: list[np.ndarray] = []
    baseline_predictions: list[np.ndarray] = []
    baseline_confidences: list[np.ndarray] = []
    finetuned_predictions: list[np.ndarray] = []
    finetuned_confidences: list[np.ndarray] = []

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            baseline_probs = torch.softmax(baseline_model(images), dim=1)
            finetuned_probs = torch.softmax(finetuned_model(images), dim=1)

            baseline_conf, baseline_pred = baseline_probs.max(dim=1)
            finetuned_conf, finetuned_pred = finetuned_probs.max(dim=1)

            labels_all.append(labels.numpy())
            baseline_predictions.append(baseline_pred.cpu().numpy())
            baseline_confidences.append(baseline_conf.cpu().numpy())
            finetuned_predictions.append(finetuned_pred.cpu().numpy())
            finetuned_confidences.append(finetuned_conf.cpu().numpy())

    del baseline_model
    del finetuned_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    labels = np.concatenate(labels_all)
    baseline_pred = np.concatenate(baseline_predictions)
    baseline_conf = np.concatenate(baseline_confidences)
    finetuned_pred = np.concatenate(finetuned_predictions)
    finetuned_conf = np.concatenate(finetuned_confidences)

    dataset_examples: Dict[int, Dict] = {}
    for dataset_index in test_indices:
        _, label = base_dataset[dataset_index]
        label = int(label)
        if label not in dataset_examples:
            dataset_examples[label] = {"dataset_index": dataset_index, "label": label}
        if len(dataset_examples) == num_classes:
            break

    candidate_by_label: Dict[int, Dict] = {}
    for local_index, dataset_index in enumerate(test_indices):
        label = int(labels[local_index])
        if int(finetuned_pred[local_index]) != label or int(baseline_pred[local_index]) == label:
            continue

        improvement_margin = float(finetuned_conf[local_index] - baseline_conf[local_index])
        current = candidate_by_label.get(label)
        if current is None or improvement_margin > current["margin"]:
            candidate_by_label[label] = {
                "dataset_index": dataset_index,
                "label": label,
                "baseline_prediction": int(baseline_pred[local_index]),
                "baseline_confidence": float(baseline_conf[local_index]),
                "finetuned_prediction": int(finetuned_pred[local_index]),
                "finetuned_confidence": float(finetuned_conf[local_index]),
                "margin": improvement_margin,
            }

    selected_examples: list[Dict] = []
    selected_labels: set[int] = set()
    for class_name in QUALITATIVE_PREFERRED_CLASSES:
        if class_name not in class_names:
            continue
        label = class_names.index(class_name)
        if label in candidate_by_label:
            selected_examples.append(candidate_by_label[label])
            selected_labels.add(label)

    remaining_candidates = sorted(
        (candidate for label, candidate in candidate_by_label.items() if label not in selected_labels),
        key=lambda candidate: candidate["margin"],
        reverse=True,
    )
    for candidate in remaining_candidates:
        if len(selected_examples) >= QUALITATIVE_SAMPLE_COUNT:
            break
        selected_examples.append(candidate)

    selected_examples = selected_examples[:QUALITATIVE_SAMPLE_COUNT]
    if not selected_examples:
        return None

    return {
        "model_key": model_key,
        "model_name": model_name,
        "class_names": class_names,
        "dataset_examples": dataset_examples,
        "selected_examples": selected_examples,
    }


def plot_dataset_mosaic(qualitative_payload: Dict, output_path: Path) -> bool:
    dataset_examples = qualitative_payload.get("dataset_examples") or {}
    class_names = qualitative_payload.get("class_names") or []
    if not dataset_examples or not class_names:
        return False

    base_dataset = _load_base_dataset(data_dir="./data")
    fig, axes = plt.subplots(2, 5, figsize=(14.6, 6.5))
    fig.patch.set_facecolor(BACKGROUND)

    for axis in axes.flat:
        axis.set_facecolor(PANEL)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#d6d3d1")
            spine.set_linewidth(1.0)

    for axis, label in zip(axes.flat, range(len(class_names))):
        sample = dataset_examples.get(label)
        if sample is None:
            axis.axis("off")
            continue
        image, _ = base_dataset[sample["dataset_index"]]
        axis.imshow(image, interpolation="nearest")
        axis.set_title(prettify_class_name(class_names[label]), fontsize=10, color=INK, pad=8)

    fig.suptitle("Real EuroSAT Test Tiles Across All 10 Classes", fontsize=16, color=INK, y=0.98)
    fig.text(
        0.5,
        0.02,
        "One held-out EuroSAT RGB tile per class from the deterministic test split.",
        ha="center",
        va="bottom",
        color=SUBTLE,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_qualitative_before_after(qualitative_payload: Dict, output_path: Path) -> bool:
    selected_examples = qualitative_payload.get("selected_examples") or []
    class_names = qualitative_payload.get("class_names") or []
    model_name = qualitative_payload.get("model_name", "Fine-tuned model")
    if not selected_examples or not class_names:
        return False

    base_dataset = _load_base_dataset(data_dir="./data")
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 10.2))
    fig.patch.set_facecolor(BACKGROUND)

    for axis in axes.flat:
        axis.set_facecolor(PANEL)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#d6d3d1")
            spine.set_linewidth(1.0)

    for axis, sample in zip(axes.flat, selected_examples):
        image, _ = base_dataset[sample["dataset_index"]]
        true_name = prettify_class_name(class_names[sample["label"]])
        baseline_name = prettify_class_name(class_names[sample["baseline_prediction"]])
        finetuned_name = prettify_class_name(class_names[sample["finetuned_prediction"]])

        axis.imshow(image, interpolation="nearest")
        axis.text(
            0.03,
            0.97,
            f"Ground truth\n{true_name}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.4,
            color=INK,
            bbox={
                "boxstyle": "round,pad=0.3",
                "fc": "#fffdf8",
                "ec": "#d6d3d1",
                "alpha": 0.96,
            },
        )
        axis.text(
            0.0,
            -0.14,
            f"Before: {baseline_name} ({sample['baseline_confidence'] * 100:.1f}%)",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            color="#b45309",
        )
        axis.text(
            0.0,
            -0.26,
            f"After:  {finetuned_name} ({sample['finetuned_confidence'] * 100:.1f}%)",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            color="#0f766e",
        )

    for axis in axes.flat[len(selected_examples):]:
        axis.axis("off")

    fig.suptitle(
        f"Before vs. After Supervised Adaptation on Real EuroSAT Test Images ({model_name})",
        fontsize=16,
        color=INK,
        y=0.98,
    )
    fig.text(
        0.5,
        0.03,
        "Representative held-out samples where the fresh EuroSAT head is wrong and the fine-tuned model recovers the correct land-use class.",
        ha="center",
        va="bottom",
        color=SUBTLE,
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95), h_pad=2.6, w_pad=1.2)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return True


def summarize_records(records: List[Dict], finetune_summary: Dict | None = None) -> Dict:
    if not records:
        return {}

    best_accuracy = max(records, key=lambda item: item["top1_accuracy"])
    fastest_cpu = min(records, key=lambda item: item["cpu_latency_ms"])
    best_size_efficiency = max(records, key=lambda item: item["efficiency_acc_per_mparam"])

    summary = {
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

    if finetune_summary and finetune_summary.get("leaderboard"):
        summary["best_finetuned_model"] = finetune_summary["leaderboard"][0]

    return summary


def copy_generated_assets(figure_paths: List[Path]) -> None:
    DOCS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for source_path in figure_paths:
        if not source_path.exists():
            continue
        destination_path = DOCS_FIGURES_DIR / source_path.name
        shutil.copy2(source_path, destination_path)


def infer_repo_links() -> dict:
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        )
        remote_url = result.stdout.strip()
    except Exception:
        return {}

    repo_url = remote_url
    if repo_url.startswith("git@github.com:"):
        repo_url = "https://github.com/" + repo_url.split(":", 1)[1]
    repo_url = repo_url.removesuffix(".git")
    if not repo_url.startswith("https://github.com/"):
        return {}

    repo_path = repo_url.replace("https://github.com/", "", 1)
    owner, repo = repo_path.split("/", 1)
    return {
        "repo_url": repo_url,
        "readme_url": f"{repo_url}/blob/main/README.md",
        "pages_url": f"https://{owner.lower()}.github.io/{repo}/",
    }


def build_finetune_site_payload(finetune_summary: Dict | None) -> dict:
    if not finetune_summary:
        return {
            "available": False,
            "completed_models": [],
            "leaderboard": [],
        }

    return {
        "available": True,
        "completed_models": sorted(list(finetune_summary.get("models", {}).keys())),
        "leaderboard": finetune_summary.get("leaderboard", []),
        "recommended_export_model": finetune_summary.get("recommended_export_model"),
        "accuracy_comparison": finetune_summary.get("accuracy_comparison", []),
    }


def build_deployment_site_payload(deployment_summary: Dict | None) -> dict:
    if not deployment_summary:
        return {"available": False}

    visible_artifacts = {}
    for key, payload in (deployment_summary.get("artifacts") or {}).items():
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status", "")).lower()
        if status in {"skipped", "pending", "missing"}:
            continue
        visible_artifacts[key] = payload

    return {
        "available": True,
        "selected_model": deployment_summary.get("selected_model"),
        "artifacts": visible_artifacts,
        "meta": deployment_summary.get("meta", {}),
    }


def build_qualitative_site_payload(qualitative_payload: Dict | None) -> dict:
    if not qualitative_payload:
        return {"available": False}

    example_classes = [
        prettify_class_name(qualitative_payload["class_names"][sample["label"]])
        for sample in qualitative_payload.get("selected_examples", [])
    ]
    model_name = qualitative_payload.get("model_name", "Fine-tuned model")
    return {
        "available": True,
        "model_name": model_name,
        "sample_count": len(qualitative_payload.get("selected_examples", [])),
        "example_classes": example_classes,
        "summary": (
            f"Real held-out EuroSAT test imagery. The dataset panel shows one test tile per class, "
            f"and the before/after panel uses {model_name} to show how supervised adaptation corrects "
            "the fresh 10-class head on representative aerial samples."
        ),
    }


def select_deployment_candidate(baseline_records: List[Dict], finetune_summary: Dict | None) -> dict | None:
    if not finetune_summary:
        return None

    leaderboard = finetune_summary.get("leaderboard", [])
    if not leaderboard:
        return None

    latency_by_model = {record["model_key"]: record["cpu_latency_ms"] for record in baseline_records}
    best_accuracy = leaderboard[0]["top1_accuracy"]
    practical_window = [
        entry for entry in leaderboard
        if latency_by_model.get(entry["model_key"]) is not None and entry["top1_accuracy"] >= best_accuracy - 1.0
    ]
    if not practical_window:
        selected = leaderboard[0]
        return {
            "model_key": selected["model_key"],
            "name": selected["name"],
            "reason": "Highest fine-tuned accuracy.",
        }

    selected = min(practical_window, key=lambda entry: latency_by_model[entry["model_key"]])
    return {
        "model_key": selected["model_key"],
        "name": selected["name"],
        "reason": "Fastest model within 1 percentage point of the top fine-tuned accuracy.",
    }


def write_site_data(
    baseline_payload: Dict,
    baseline_records: List[Dict],
    figure_paths: dict,
    finetune_summary: Dict | None,
    deployment_summary: Dict | None,
    qualitative_payload: Dict | None,
) -> None:
    summary = summarize_records(baseline_records, finetune_summary=finetune_summary)
    meta = extract_meta(baseline_payload)
    links = infer_repo_links()
    deployment_candidate = select_deployment_candidate(baseline_records, finetune_summary)
    if PDF_OUTPUT_FILE.exists():
        links["pdf_path"] = f"assets/{PDF_OUTPUT_FILE.name}"

    site_payload = {
        "generated_at": meta.get("timestamp"),
        "meta": meta,
        "summary": summary,
        "models": baseline_records,
        "figures": {
            "accuracy_vs_latency": f"assets/figures/{figure_paths['accuracy_vs_latency'].name}",
            "parameter_footprint": f"assets/figures/{figure_paths['parameter_footprint'].name}",
            "latency_breakdown": f"assets/figures/{figure_paths['latency_breakdown'].name}",
            "finetune_accuracy_delta": (
                f"assets/figures/{figure_paths['finetune_accuracy_delta'].name}"
                if figure_paths.get("finetune_accuracy_delta")
                else None
            ),
            "dataset_mosaic": (
                f"assets/figures/{figure_paths['dataset_mosaic'].name}"
                if figure_paths.get("dataset_mosaic")
                else None
            ),
            "qualitative_before_after": (
                f"assets/figures/{figure_paths['qualitative_before_after'].name}"
                if figure_paths.get("qualitative_before_after")
                else None
            ),
        },
        "finetune": build_finetune_site_payload(finetune_summary),
        "deployment": build_deployment_site_payload(deployment_summary),
        "qualitative": build_qualitative_site_payload(qualitative_payload),
        "links": links,
    }
    if deployment_candidate:
        site_payload["finetune"]["deployment_candidate"] = deployment_candidate

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
    finetune_summary = load_optional_json(FINETUNE_RESULTS_FILE)
    deployment_summary = load_optional_json(DEPLOYMENT_RESULTS_FILE)
    records = normalize_model_records(benchmark_payload, experiment_name="baseline")

    if not records:
        print("  ERROR: No complete model records were found in results/benchmark_results.json")
        sys.exit(1)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    figure_paths = {
        "accuracy_vs_latency": FIGURES_DIR / "accuracy_vs_latency.png",
        "parameter_footprint": FIGURES_DIR / "parameter_footprint.png",
        "latency_breakdown": FIGURES_DIR / "cpu_gpu_latency_breakdown.png",
        "finetune_accuracy_delta": FIGURES_DIR / "baseline_vs_finetuned_accuracy.png",
        "dataset_mosaic": FIGURES_DIR / "eurosat_testset_mosaic.png",
        "qualitative_before_after": FIGURES_DIR / "qualitative_before_after.png",
    }
    stale_radar = FIGURES_DIR / "tradeoff_radar.png"

    benchmark_meta = extract_meta(benchmark_payload)
    print(f"  Dataset   : {benchmark_meta.get('dataset', 'N/A')}")
    print(f"  Protocol  : {benchmark_meta.get('protocol_label', 'N/A')}")
    print(f"  Models    : {len(records)}")
    print()

    print("  [1]  Accuracy vs. latency scatter ...          ", end="", flush=True)
    plot_accuracy_vs_latency(records, figure_paths["accuracy_vs_latency"])
    print(f"OK  {figure_paths['accuracy_vs_latency']}")

    print("  [2]  Parameter footprint plot ...             ", end="", flush=True)
    plot_parameter_footprint(records, figure_paths["parameter_footprint"])
    print(f"OK  {figure_paths['parameter_footprint']}")

    print("  [3]  CPU/GPU latency comparison ...           ", end="", flush=True)
    plot_latency_breakdown(records, figure_paths["latency_breakdown"])
    print(f"OK  {figure_paths['latency_breakdown']}")

    generated_figures = [
        figure_paths["accuracy_vs_latency"],
        figure_paths["parameter_footprint"],
        figure_paths["latency_breakdown"],
    ]

    if plot_finetune_accuracy_delta(records, finetune_summary, figure_paths["finetune_accuracy_delta"]):
        generated_figures.append(figure_paths["finetune_accuracy_delta"])
        print(f"  [4]  Fine-tune accuracy comparison ...        OK  {figure_paths['finetune_accuracy_delta']}")
    else:
        figure_paths["finetune_accuracy_delta"] = None

    qualitative_payload = build_qualitative_examples(
        benchmark_payload,
        finetune_summary,
        deployment_summary,
        data_dir="./data",
    )

    print("  [5]  EuroSAT class mosaic ...                 ", end="", flush=True)
    if qualitative_payload and plot_dataset_mosaic(qualitative_payload, figure_paths["dataset_mosaic"]):
        generated_figures.append(figure_paths["dataset_mosaic"])
        print(f"OK  {figure_paths['dataset_mosaic']}")
    else:
        figure_paths["dataset_mosaic"] = None
        print("SKIPPED")

    print("  [6]  Qualitative before/after panel ...      ", end="", flush=True)
    if qualitative_payload and plot_qualitative_before_after(qualitative_payload, figure_paths["qualitative_before_after"]):
        generated_figures.append(figure_paths["qualitative_before_after"])
        print(f"OK  {figure_paths['qualitative_before_after']}")
    else:
        figure_paths["qualitative_before_after"] = None
        print("SKIPPED")

    if stale_radar.exists():
        stale_radar.unlink()
        print("  [x]  Removed stale radar chart from previous output.")

    copy_generated_assets(generated_figures)
    write_site_data(
        benchmark_payload,
        records,
        figure_paths,
        finetune_summary,
        deployment_summary,
        qualitative_payload,
    )

    print()
    print("  ------------------------------------------------")
    print("  Figures written to results/figures/")
    print("  Dashboard data written to docs/site_data.json and docs/site_data.js")
    print("  ------------------------------------------------")
    print()


if __name__ == "__main__":
    main()
