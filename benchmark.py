#!/usr/bin/env python3
"""
Efficient Vision Model Benchmark
=================================
Main entry point for benchmarking lightweight vision models.

Benchmarks each model in the registry on CIFAR-10, measuring:
  - Top-1 / Top-5 accuracy
  - Inference latency (CPU)
  - Model size and parameter count
  - Estimated FLOPs
  - Throughput at various batch sizes

Usage:
    python benchmark.py                      # Benchmark all models
    python benchmark.py --model mobilenet_v2  # Single model
    python benchmark.py --list                # List available models
"""

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
from tabulate import tabulate

from src.models import load_model_config, get_model, get_model_info
from src.data import get_cifar10_loaders
from src.timing import measure_latency, measure_throughput
from src.metrics import evaluate_accuracy, estimate_flops


# ═══════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════

RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "benchmark_results.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def benchmark_single_model(
    model_key: str,
    test_loader,
    device: str = "cpu",
    input_size: int = 224,
) -> dict:
    """Run full benchmark suite for a single model."""
    print(f"\n{'═' * 60}")
    print(f"  Benchmarking: {model_key}")
    print(f"{'═' * 60}")

    # Load model
    print("  → Loading model...")
    model = get_model(model_key, num_classes=10)
    model_info = get_model_info(model)
    print(f"    Parameters: {model_info['param_str']} ({model_info['size_mb']} MB)")

    # FLOPs
    print("  → Estimating FLOPs...")
    flops_info = estimate_flops(model, input_size)
    if flops_info["flops_str"] != "N/A":
        print(f"    FLOPs: {flops_info['flops_str']}")

    # Latency (CPU)
    print("  → Measuring latency (CPU, batch=1)...")
    latency_cpu = measure_latency(
        model, input_size=input_size, device="cpu",
        warmup_runs=5, benchmark_runs=50, batch_size=1
    )
    print(f"    Latency: {latency_cpu['mean_ms']:.1f} ± {latency_cpu['std_ms']:.1f} ms")

    # GPU latency (if available)
    latency_gpu = None
    if device == "cuda":
        print("  → Measuring latency (GPU, batch=1)...")
        latency_gpu = measure_latency(
            model, input_size=input_size, device="cuda",
            warmup_runs=10, benchmark_runs=100, batch_size=1
        )
        print(f"    Latency: {latency_gpu['mean_ms']:.1f} ± {latency_gpu['std_ms']:.1f} ms")

    # Throughput
    print("  → Measuring throughput...")
    throughput = measure_throughput(
        model, input_size=input_size, device=device,
        batch_sizes=[1, 16, 64], duration_seconds=3.0
    )
    print(f"    Throughput (bs=1): {throughput.get(1, 'N/A')} img/s")

    # Accuracy on CIFAR-10
    print("  → Evaluating accuracy on CIFAR-10 test set...")
    accuracy = evaluate_accuracy(model, test_loader, device=device)
    print(f"    Top-1: {accuracy['top1_accuracy']:.2f}%")
    print(f"    Top-5: {accuracy['top5_accuracy']:.2f}%")

    # Compile results
    result = {
        "model_key": model_key,
        "model_info": model_info,
        "flops": flops_info,
        "latency_cpu": latency_cpu,
        "latency_gpu": latency_gpu,
        "throughput": {str(k): v for k, v in throughput.items()},
        "accuracy": accuracy,
    }

    print(f"  ✓ Done: {model_key}")
    return result


def run_benchmark(model_keys: list, device: str = "cpu"):
    """Run benchmarks for all specified models."""
    print("\n╔══════════════════════════════════════════════╗")
    print("║   Efficient Vision Model Benchmark           ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"\nDevice: {device.upper()}")
    print(f"Models: {len(model_keys)}")
    print(f"Dataset: CIFAR-10 (10,000 test images)")

    # Load data once
    print("\n→ Loading CIFAR-10 dataset...")
    _, test_loader = get_cifar10_loaders(
        input_size=224, batch_size=64, num_workers=2
    )

    # Benchmark each model
    all_results = {}
    for key in model_keys:
        result = benchmark_single_model(key, test_loader, device)
        all_results[key] = result

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "models": all_results,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n✓ Results saved to {RESULTS_FILE}")

    # Print summary table
    print_summary_table(all_results)

    return all_results


def print_summary_table(results: dict):
    """Print a formatted comparison table."""
    headers = [
        "Model", "Params", "Size (MB)", "FLOPs",
        "Latency (ms)", "Top-1 (%)", "Top-5 (%)"
    ]
    rows = []

    for key, data in results.items():
        rows.append([
            data["model_info"]["param_str"],
            key,
            data["model_info"]["size_mb"],
            data["flops"]["flops_str"],
            data["latency_cpu"]["mean_ms"],
            data["accuracy"]["top1_accuracy"],
            data["accuracy"]["top5_accuracy"],
        ])

    print(f"\n{'═' * 80}")
    print("  BENCHMARK SUMMARY")
    print(f"{'═' * 80}")
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline", floatfmt=".2f"))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Efficient Vision Model Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Benchmark a specific model (key from configs/models.yaml)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--device", type=str, default=DEVICE,
        choices=["cpu", "cuda"],
        help=f"Device to run on (default: {DEVICE})"
    )
    args = parser.parse_args()

    config = load_model_config()

    if args.list:
        from src.models import list_available_models
        list_available_models()
        sys.exit(0)

    if args.model:
        if args.model not in config:
            print(f"Error: Unknown model '{args.model}'")
            print(f"Available: {', '.join(config.keys())}")
            sys.exit(1)
        model_keys = [args.model]
    else:
        model_keys = list(config.keys())

    run_benchmark(model_keys, device=args.device)


if __name__ == "__main__":
    main()
