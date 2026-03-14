#!/usr/bin/env python3
"""
02_generate_visualizations.py — Pareto Trade-off Charts
=======================================================
Reads benchmark telemetry from results/benchmark_results.json
and generates publication-quality visualizations.

Run:  python 02_generate_visualizations.py
      (No CLI arguments needed)

Author: Kenan Radheshyam Trivedi
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────
RESULTS_FILE = Path("results/benchmark_results.json")
FIGURES_DIR = Path("results/figures")

# ─── Visual Theme (dark, modern) ─────────────────────────
sns.set_theme(style="darkgrid", context="talk")
plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#0f1117",
    "axes.edgecolor": "#2a2e39",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
})

PALETTE = ["#58a6ff", "#3fb950", "#d2a8ff", "#f97583",
           "#79c0ff", "#56d364", "#e3b341", "#ff7b72", "#a5d6ff"]


def load_data() -> dict:
    """Load the benchmark JSON payload."""
    if not RESULTS_FILE.exists():
        print(f"\n  ✗ File not found: {RESULTS_FILE}")
        print("    Run  python 01_run_benchmark.py  first.\n")
        sys.exit(1)
    with open(RESULTS_FILE) as f:
        return json.load(f)


# ┌─────────────────────────────────────────────────────────┐
# │  Chart 1 — Accuracy vs. CPU Latency (Pareto Scatter)   │
# └─────────────────────────────────────────────────────────┘

def plot_accuracy_vs_latency(models: dict, out: Path):
    fig, ax = plt.subplots(figsize=(11, 7))

    names, lats, accs, sizes = [], [], [], []
    for key, m in models.items():
        lat = m.get("latency_cpu", {}).get("median_ms")
        acc = m.get("accuracy", {}).get("top1_accuracy")
        sz = m.get("model_info", {}).get("size_mb", 10)
        if lat is None or acc is None:
            continue
        names.append(m.get("name", key))
        lats.append(lat)
        accs.append(acc)
        sizes.append(sz)

    if not names:
        print("    ⓘ  Not enough data for scatter plot.")
        return

    sizes_scaled = [max(s * 12, 80) for s in sizes]

    for i, (x, y, s, name) in enumerate(zip(lats, accs, sizes_scaled, names)):
        ax.scatter(x, y, s=s, color=PALETTE[i % len(PALETTE)],
                   alpha=0.85, edgecolors="white", linewidths=0.8,
                   zorder=3, label=name)
        ax.annotate(name, (x, y), xytext=(8, -4),
                    textcoords="offset points", fontsize=9,
                    color="#c9d1d9", fontweight="bold")

    ax.set_xlabel("CPU Inference Latency (ms)  →  lower is better", fontsize=12)
    ax.set_ylabel("EuroSAT Aerial Top-1 Accuracy (%)  →  higher is better", fontsize=12)
    ax.set_title("Edge Deployment Trade-off: Accuracy vs. Latency\n(EuroSAT Aerial · Pretrained · 2024-2025 SOTA)",
                 fontsize=14, pad=14)
    ax.legend(loc="lower right", fontsize=8, frameon=True, ncol=2)

    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ┌─────────────────────────────────────────────────────────┐
# │  Chart 2 — Model Size Comparison (Horizontal Bar)      │
# └─────────────────────────────────────────────────────────┘

def plot_model_sizes(models: dict, out: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    names, params = [], []
    for key, m in models.items():
        p = m.get("model_info", {}).get("total_params")
        if p is None:
            continue
        names.append(m.get("name", key))
        params.append(p / 1e6)

    if not names:
        return

    # Sort by param count
    order = np.argsort(params)
    names = [names[i] for i in order]
    params = [params[i] for i in order]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]

    bars = ax.barh(names, params, color=colors, edgecolor="#30363d", height=0.6)
    for bar, val in zip(bars, params):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}M", va="center", fontsize=10, color="#c9d1d9")

    ax.set_xlabel("Parameters (Millions)", fontsize=12)
    ax.set_title("Model Size Comparison", fontsize=14, pad=12)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0fM"))

    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ┌─────────────────────────────────────────────────────────┐
# │  Main                                                   │
# └─────────────────────────────────────────────────────────┘

def main():
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║   VISUALIZATION GENERATOR                              ║")
    print("  ║   Building trade-off charts from benchmark telemetry   ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()

    data = load_data()
    models = data.get("models", {})
    meta = data.get("meta", {})

    print(f"  Dataset  : {meta.get('dataset', 'N/A')}")
    print(f"  Device   : {meta.get('device', 'N/A')}")
    print(f"  Models   : {len(models)}")
    print()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Chart 1
    p1 = FIGURES_DIR / "accuracy_vs_latency.png"
    print(f"  [1]  Generating Accuracy vs. Latency scatter ...  ", end="", flush=True)
    plot_accuracy_vs_latency(models, p1)
    print(f"✓  {p1}")

    # Chart 2
    p2 = FIGURES_DIR / "model_size_comparison.png"
    print(f"  [2]  Generating Model Size bar chart ...           ", end="", flush=True)
    plot_model_sizes(models, p2)
    print(f"✓  {p2}")

    print()
    print("  ──────────────────────────────────────────────")
    print("  All figures saved to results/figures/")
    print("  ──────────────────────────────────────────────")
    print()


if __name__ == "__main__":
    main()
