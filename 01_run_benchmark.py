#!/usr/bin/env python3
"""
01_run_benchmark.py
===================
Interactive benchmarking suite for computer vision models.
Evaluates accuracy, latency, and memory requirements using
PyTorch and Hugging Face's `timm` library.

Run directly via python: `python 01_run_benchmark.py`
(No CLI arguments required. Interactive menu will guide you).
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import torch
from tabulate import tabulate

from src.models import load_model_config, get_model, get_model_info, list_available_models
from src.data import get_cifar10_loaders
from src.timing import measure_latency, measure_throughput
from src.metrics import evaluate_accuracy, estimate_flops

RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "benchmark_results.json"


def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    clear_terminal()
    print("======================================================")
    print(" 🚀 EFFICIENT VISION MODEL BENCHMARK                   ")
    print("======================================================")
    print(" Evaluates cutting-edge (2025) edge architectures and  ")
    print(" classic baselines for deployment trade-offs.          ")
    print("------------------------------------------------------\n")


def prompt_device() -> str:
    """Interactively select hardware execution target."""
    print("[1] CUDA (NVIDIA GPU - Fastest)")
    print("[2] CPU  (General Compute)")
    
    if not torch.cuda.is_available():
        print("\nNote: CUDA is completely unavailable on this machine. Forcing CPU.")
        return "cpu"

    while True:
        choice = input("Select execution device [1/2]: ").strip()
        if choice == '1':
            return "cuda"
        elif choice == '2':
            return "cpu"
        print("Invalid choice, please select 1 or 2.")


def prompt_models(config: dict) -> list:
    """Interactively select which models to benchmark."""
    print("\nAvailable Architectures:")
    print("------------------------")
    keys = list(config.keys())
    for i, key in enumerate(keys, 1):
        src = config[key].get('source', 'torchvision')
        print(f"  [{i}] {key:<18} ({src})")
    
    print(f"  [{len(keys)+1}] Run ALL Models (Patience required)")
    
    while True:
        try:
            choice = input(f"\nSelect model to benchmark (1-{len(keys)+1}): ").strip()
            idx = int(choice)
            if 1 <= idx <= len(keys):
                return [keys[idx-1]]
            elif idx == len(keys) + 1:
                return keys
            print("Invalid range.")
        except ValueError:
            print("Please enter a valid number.")


def run_benchmark_pipeline(model_keys: list, device: str):
    """Execution engine for benchmarking models individually."""
    print("\n======================================================")
    print(f" → Executing Benchmark on [{device.upper()}] for {len(model_keys)} models...")
    print("======================================================\n")

    print("[*] Automatically pre-loading EuroSAT Aerial Imagery (Auto-downloading if missing)...")
    from src.data import get_eurosat_loaders
    _, test_loader = get_eurosat_loaders(input_size=224, batch_size=64, num_workers=2)

    all_results = {}
    for i, model_key in enumerate(model_keys, 1):
        print(f"\n--- [Model {i}/{len(model_keys)}]: {model_key} ---")
        
        # Load Architecture
        print("  1. Pulling weights (timm/torchvision) & adapting classification heads...")
        model = get_model(model_key, num_classes=10)
        model_info = get_model_info(model)
        
        # Structure Memory & FLOPs footprint
        print(f"  2. Analyzing memory footprint ({model_info['param_str']} Params)...")
        in_size = load_model_config()[model_key].get('input_size', 224)
        flops_info = estimate_flops(model, in_size)
        
        # CPU Latency Profile
        print("  3. Profiling latency edge-bound (CPU)...")
        latency_cpu = measure_latency(
            model, input_size=in_size, device="cpu",
            warmup_runs=5, benchmark_runs=30, batch_size=1
        )
        
        # GPU Latency Profile (Conditional)
        latency_gpu = None
        if device == "cuda":
            print("  4. High-performance profile (GPU Latency)...")
            latency_gpu = measure_latency(
                model, input_size=in_size, device="cuda",
                warmup_runs=10, benchmark_runs=50, batch_size=1
            )
        
        # Measure throughput
        print("  5. Benchmarking variable batch scaling throughput...")
        throughput = measure_throughput(
            model, input_size=in_size, device=device,
            batch_sizes=[1, 16], duration_seconds=2.0
        )
        
        # Accuracy Execution
        print("  6. Running standard aerial dataset evaluation loop...")
        accuracy = evaluate_accuracy(model, test_loader, device=device)
        
        print(f"  => Complete! {accuracy['top1_accuracy']}% Top-1 ACC 🚀")
        
        # Store execution artifacts
        all_results[model_key] = {
            "model_key": model_key,
            "model_info": model_info,
            "flops": flops_info,
            "latency_cpu": latency_cpu,
            "latency_gpu": latency_gpu,
            "throughput": {str(k): v for k, v in throughput.items()},
            "accuracy": accuracy,
        }

    # Save artifact payload 
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "models": all_results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(payload, f, indent=2, default=str)
        
    print(f"\n✅ Pipeline Complete. All telemetry saved locally to '{RESULTS_FILE}'")
    
    print("\nWould you like to generate the resulting Pareto & Radar visuals now?")
    do_viz = input("[Y/N]: ").strip().lower()
    if do_viz == 'y':
        # Execute the visualization generation script natively via python
        os.system(f"{sys.executable} 02_generate_visualizations.py --silent")
        print("\n=> Visualizations generated successfully under `results/figures/`.")


def main():
    print_header()
    config = load_model_config()
    device = prompt_device()
    models_to_run = prompt_models(config)
    run_benchmark_pipeline(models_to_run, device)


if __name__ == '__main__':
    main()
