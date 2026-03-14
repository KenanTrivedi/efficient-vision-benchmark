#!/usr/bin/env python3
"""
Visualization suite for benchmark results.

Generates comparison plots:
  1. Accuracy vs. Latency scatter (Pareto front)
  2. Model size comparison (bar chart)
  3. Trade-off radar chart (multi-dimensional view)

Usage:
    python visualize.py
    python visualize.py --results results/benchmark_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

# ═══════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════

RESULTS_FILE = Path("results/benchmark_results.json")
FIGURES_DIR = Path("results/figures")

# Color palette — dark theme, professional
COLORS = {
    "resnet18": "#4fc3f7",
    "mobilenet_v2": "#81c784",
    "mobilenet_v3_small": "#aed581",
    "efficientnet_b0": "#ffb74d",
    "shufflenet_v2_x1_0": "#f06292",
}
DEFAULT_COLOR = "#90a4ae"

DARK_BG = "#0a0e17"
CARD_BG = "#141b2d"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#1e2a3a"
ACCENT = "#00e5ff"


def setup_dark_style():
    """Configure matplotlib for dark theme plots."""
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor": CARD_BG,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.size": 11,
    })


def load_results(path: Path) -> dict:
    """Load benchmark results from JSON."""
    if not path.exists():
        print(f"Error: Results file not found: {path}")
        print("Run 'python benchmark.py' first.")
        sys.exit(1)

    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════
# Plot 1: Accuracy vs. Latency
# ═══════════════════════════════════════════════════

def plot_accuracy_vs_latency(results: dict, save_path: Path):
    """
    Scatter plot of Top-1 accuracy vs inference latency.
    Ideal models are in the top-left (high accuracy, low latency).
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for key, data in results["models"].items():
        acc = data["accuracy"]["top1_accuracy"]
        lat = data["latency_cpu"]["mean_ms"]
        size_mb = data["model_info"]["size_mb"]
        color = COLORS.get(key, DEFAULT_COLOR)

        # Bubble size proportional to model size
        bubble_size = max(size_mb * 15, 80)

        ax.scatter(
            lat, acc, s=bubble_size, c=color, alpha=0.85,
            edgecolors="white", linewidths=1.5, zorder=5,
        )
        ax.annotate(
            key.replace("_", "\n"),
            (lat, acc),
            textcoords="offset points",
            xytext=(12, 8),
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    ax.set_xlabel("Inference Latency (ms) — CPU, batch=1", fontsize=12, fontweight="bold")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Accuracy vs. Latency Trade-off", fontsize=16, fontweight="bold", pad=20)
    ax.grid(True, alpha=0.2)

    # Add annotation for "ideal" region
    ax.annotate(
        "← Better",
        xy=(0.02, 0.98), xycoords="axes fraction",
        fontsize=10, color=ACCENT, alpha=0.5,
        verticalalignment="top",
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ═══════════════════════════════════════════════════
# Plot 2: Model Size Comparison
# ═══════════════════════════════════════════════════

def plot_model_size(results: dict, save_path: Path):
    """Horizontal bar chart comparing parameter counts and sizes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    names = []
    params = []
    sizes = []
    colors = []

    for key, data in results["models"].items():
        names.append(key)
        params.append(data["model_info"]["total_params"] / 1e6)
        sizes.append(data["model_info"]["size_mb"])
        colors.append(COLORS.get(key, DEFAULT_COLOR))

    # Sort by params
    order = np.argsort(params)
    names = [names[i] for i in order]
    params = [params[i] for i in order]
    sizes = [sizes[i] for i in order]
    colors = [colors[i] for i in order]

    y_pos = np.arange(len(names))

    # Panel 1: Parameter count
    bars1 = ax1.barh(y_pos, params, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=10)
    ax1.set_xlabel("Parameters (Millions)", fontsize=11, fontweight="bold")
    ax1.set_title("Parameter Count", fontsize=13, fontweight="bold")
    ax1.grid(axis="x", alpha=0.2)

    for bar, val in zip(bars1, params):
        ax1.text(
            bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}M", va="center", fontsize=9, color=TEXT_COLOR,
        )

    # Panel 2: Disk size
    bars2 = ax2.barh(y_pos, sizes, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=10)
    ax2.set_xlabel("Size (MB)", fontsize=11, fontweight="bold")
    ax2.set_title("Model Size on Disk", fontsize=13, fontweight="bold")
    ax2.grid(axis="x", alpha=0.2)

    for bar, val in zip(bars2, sizes):
        ax2.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f} MB", va="center", fontsize=9, color=TEXT_COLOR,
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ═══════════════════════════════════════════════════
# Plot 3: Trade-off Radar Chart
# ═══════════════════════════════════════════════════

def plot_tradeoff_radar(results: dict, save_path: Path):
    """
    Radar chart showing normalized trade-off dimensions:
    accuracy, speed (1/latency), compactness (1/params), efficiency (1/FLOPs).
    """
    categories = ["Accuracy", "Speed", "Compactness", "Efficiency"]
    N = len(categories)

    # Collect raw values
    model_keys = list(results["models"].keys())
    raw_data = {}

    for key in model_keys:
        data = results["models"][key]
        acc = data["accuracy"]["top1_accuracy"]
        latency = data["latency_cpu"]["mean_ms"]
        params = data["model_info"]["total_params"]
        flops = data["flops"].get("flops") or 1e9  # fallback

        raw_data[key] = [acc, 1.0 / latency, 1.0 / params, 1.0 / flops]

    # Normalize each dimension to [0, 1]
    all_values = np.array(list(raw_data.values()))
    mins = all_values.min(axis=0)
    maxs = all_values.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # avoid division by zero

    normalized = {}
    for key in model_keys:
        normalized[key] = ((np.array(raw_data[key]) - mins) / ranges).tolist()

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor(CARD_BG)
    fig.patch.set_facecolor(DARK_BG)

    for key in model_keys:
        values = normalized[key] + normalized[key][:1]
        color = COLORS.get(key, DEFAULT_COLOR)
        ax.plot(angles, values, linewidth=2, color=color, label=key)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold", color=TEXT_COLOR)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color=TEXT_COLOR)
    ax.grid(color=GRID_COLOR, alpha=0.3)

    ax.legend(
        loc="upper right", bbox_to_anchor=(1.3, 1.1),
        fontsize=9, framealpha=0.3,
        facecolor=CARD_BG, edgecolor=GRID_COLOR,
    )

    ax.set_title(
        "Multi-dimensional Trade-off Comparison",
        fontsize=14, fontweight="bold", pad=30, color=TEXT_COLOR,
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark visualizations")
    parser.add_argument(
        "--results", type=str, default=str(RESULTS_FILE),
        help="Path to benchmark results JSON"
    )
    args = parser.parse_args()

    setup_dark_style()

    results = load_results(Path(args.results))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n╔══════════════════════════════════════════════╗")
    print("║   Generating Benchmark Visualizations         ║")
    print("╚══════════════════════════════════════════════╝\n")

    plot_accuracy_vs_latency(results, FIGURES_DIR / "accuracy_vs_latency.png")
    plot_model_size(results, FIGURES_DIR / "model_size_comparison.png")
    plot_tradeoff_radar(results, FIGURES_DIR / "tradeoff_radar.png")

    print(f"\n✓ All figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
