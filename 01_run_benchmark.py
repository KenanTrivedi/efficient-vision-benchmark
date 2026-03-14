#!/usr/bin/env python3
"""
01_run_benchmark.py — Aerial Vision Model Benchmark
====================================================
Evaluates state-of-the-art lightweight edge vision architectures
on the EuroSAT aerial/satellite imagery dataset.

Measures: Accuracy · Latency · FLOPs · Throughput · Memory Footprint

Run:  python 01_run_benchmark.py
      (No CLI arguments needed — fully interactive)

Author: Kenan Radheshyam Trivedi
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch

from src.models import load_model_config, get_model, get_model_info
from src.data import get_eurosat_loaders
from src.timing import measure_latency, measure_throughput
from src.metrics import evaluate_accuracy, estimate_flops

# ─── Paths ────────────────────────────────────────────────
RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "benchmark_results.json"


# ┌─────────────────────────────────────────────────────────┐
# │  Terminal UI Helpers                                    │
# └─────────────────────────────────────────────────────────┘

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def banner():
    clear()
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║                                                        ║")
    print("  ║   AERIAL VISION MODEL BENCHMARK                        ║")
    print("  ║   Evaluating Edge-Optimized Architectures on EuroSAT   ║")
    print("  ║                                                        ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()
    hw = "CUDA — " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    print(f"  Hardware : {hw}")
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  Time     : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()


def section(title: str):
    width = 58
    print()
    print(f"  ┌{'─' * width}┐")
    print(f"  │  {title:<{width - 2}}│")
    print(f"  └{'─' * width}┘")
    print()


def step(number: int, text: str):
    print(f"    [{number}] {text}")


def result_line(label: str, value: str):
    print(f"        {label:<26} {value}")


# ┌─────────────────────────────────────────────────────────┐
# │  Interactive Device Selection                           │
# └─────────────────────────────────────────────────────────┘

def select_device() -> str:
    print("  Select compute device:")
    print("    [1]  NVIDIA CUDA  (GPU — recommended for full benchmark)")
    print("    [2]  CPU          (slower but always available)")
    print()

    if not torch.cuda.is_available():
        print("  ⓘ  CUDA not detected. Falling back to CPU.")
        return "cpu"

    while True:
        choice = input("  Your choice [1/2]: ").strip()
        if choice == "1":
            return "cuda"
        if choice == "2":
            return "cpu"
        print("  Please enter 1 or 2.")


# ┌─────────────────────────────────────────────────────────┐
# │  Interactive Model Selection                            │
# └─────────────────────────────────────────────────────────┘

def select_models(config: dict) -> list:
    keys = list(config.keys())
    print("  Available architectures:")
    print()
    for i, key in enumerate(keys, 1):
        cfg = config[key]
        src = cfg.get("source", "torchvision")
        yr = cfg.get("year", "—")
        print(f"    [{i:>2}]  {cfg['name']:<22} {src:<12} ({yr})")

    print(f"    [{len(keys) + 1:>2}]  ── Run ALL models ──")
    print()

    while True:
        try:
            choice = int(input("  Select model number: ").strip())
            if 1 <= choice <= len(keys):
                return [keys[choice - 1]]
            if choice == len(keys) + 1:
                return keys
        except ValueError:
            pass
        print("  Invalid selection. Try again.")


# ┌─────────────────────────────────────────────────────────┐
# │  Core Benchmark Pipeline                                │
# └─────────────────────────────────────────────────────────┘

def run_benchmark(model_keys: list, device: str):
    t0 = time.time()

    # ── Data ──────────────────────────────────────────────
    section("Loading EuroSAT Aerial Dataset")
    step(1, "Downloading / caching EuroSAT (27,000 satellite images, 10 land-use classes)...")
    _, test_loader = get_eurosat_loaders(input_size=224, batch_size=64, num_workers=2)
    step(2, f"Dataset ready — {len(test_loader.dataset):,} images loaded.")

    all_results = {}

    for idx, key in enumerate(model_keys, 1):
        cfg = load_model_config()[key]
        section(f"Model {idx}/{len(model_keys)} — {cfg['name']}")
        print(f"    Paper : {cfg.get('paper', 'N/A')}")
        print(f"    Source: {cfg.get('source', 'torchvision')} | Input: {cfg.get('input_size', 224)}px")
        print()

        try:
            in_size = cfg.get("input_size", 224)

            # 1. Load
            step(1, "Loading pretrained weights & adapting classification head (10 classes)...")
            model = get_model(key, num_classes=10)
            info = get_model_info(model)
            result_line("Parameters", info["param_str"])
            result_line("Disk footprint", f"{info['size_mb']} MB")

            # 2. FLOPs
            step(2, "Estimating computational cost (MACs / FLOPs)...")
            flops = estimate_flops(model, in_size)
            result_line("MACs", flops.get("macs_str", "N/A"))
            result_line("FLOPs", flops.get("flops_str", "N/A"))

            # 3. CPU latency
            step(3, "Profiling CPU inference latency (20 runs, 5 warmup)...")
            lat_cpu = measure_latency(model, in_size, "cpu", warmup_runs=5, benchmark_runs=20, batch_size=1)
            result_line("CPU latency (median)", f"{lat_cpu['median_ms']} ms")

            # 4. GPU latency
            lat_gpu = None
            if device == "cuda":
                step(4, "Profiling GPU inference latency (30 runs, 5 warmup)...")
                lat_gpu = measure_latency(model, in_size, "cuda", warmup_runs=5, benchmark_runs=30, batch_size=1)
                result_line("GPU latency (median)", f"{lat_gpu['median_ms']} ms")

            # 5. Throughput
            step(5, "Measuring batch throughput (batch sizes 1, 16)...")
            throughput = measure_throughput(model, in_size, device, batch_sizes=[1, 16], duration_seconds=1.0)
            for bs, ips in throughput.items():
                result_line(f"Throughput (bs={bs})", f"{ips} img/s")

            # 6. Accuracy
            step(6, "Evaluating accuracy on EuroSAT aerial test set...")
            acc = evaluate_accuracy(model, test_loader, device=device)
            result_line("Top-1 accuracy", f"{acc['top1_accuracy']}%")
            result_line("Top-5 accuracy", f"{acc['top5_accuracy']}%")

            print(f"\n    ✓ {cfg['name']} complete.")

            all_results[key] = {
                "name": cfg["name"],
                "year": cfg.get("year"),
                "paper": cfg.get("paper"),
                "model_info": info,
                "flops": flops,
                "latency_cpu": lat_cpu,
                "latency_gpu": lat_gpu,
                "throughput": {str(k): v for k, v in throughput.items()},
                "accuracy": acc,
            }
        except Exception as e:
            print(f"\n    ✗ Error benchmarking {key}: {e}")

    # ── Save ──────────────────────────────────────────────
    section("Saving Results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "device": device,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "dataset": "EuroSAT",
            "num_images": len(test_loader.dataset),
        },
        "models": all_results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    elapsed = time.time() - t0
    step(1, f"Results saved to {RESULTS_FILE}")
    step(2, f"Total benchmark time: {elapsed / 60:.1f} minutes")
    print()
    print("  ──────────────────────────────────────────────")
    print("  Next step:  python 02_generate_visualizations.py")
    print("  ──────────────────────────────────────────────")
    print()


# ┌─────────────────────────────────────────────────────────┐
# │  Main                                                   │
# └─────────────────────────────────────────────────────────┘

def main():
    banner()
    config = load_model_config()
    device = select_device()
    models = select_models(config)
    run_benchmark(models, device)


if __name__ == "__main__":
    main()
