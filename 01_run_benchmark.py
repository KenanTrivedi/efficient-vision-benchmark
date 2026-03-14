#!/usr/bin/env python3
"""
01_run_benchmark.py - EuroSAT transfer baseline benchmark
==========================================================
Evaluates pretrained vision backbones after resetting their classifier
heads to EuroSAT's 10 classes. This is a head-reset transfer baseline,
not a true zero-shot evaluation.

Measures: Accuracy | Latency | FLOPs | Throughput | Memory Footprint
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch

from src.data import DEFAULT_SEED, get_eurosat_loaders
from src.metrics import estimate_flops, evaluate_accuracy
from src.models import get_model, get_model_info, load_model_config
from src.timing import measure_latency, measure_throughput


RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "benchmark_results.json"
EXPERIMENT_PROTOCOL = "head_reset_transfer_baseline"


def clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def banner() -> None:
    clear()
    print()
    print("  ==========================================================")
    print("    EUROSAT TRANSFER BASELINE BENCHMARK")
    print("    Pretrained backbones with reset 10-class heads")
    print("  ==========================================================")
    print()
    hardware = "CUDA - " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    print(f"  Hardware : {hardware}")
    print(f"  PyTorch  : {torch.__version__}")
    print(f"  Time     : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()


def section(title: str) -> None:
    print()
    print(f"  --- {title} {'-' * max(0, 52 - len(title))}")
    print()


def step(number: int, text: str) -> None:
    print(f"    [{number}] {text}")


def result_line(label: str, value: str) -> None:
    print(f"        {label:<26} {value}")


def select_device() -> str:
    print("  Select compute device:")
    print("    [1]  NVIDIA CUDA  (GPU - recommended for full benchmark)")
    print("    [2]  CPU          (slower but always available)")
    print()

    if not torch.cuda.is_available():
        print("  INFO: CUDA not detected. Falling back to CPU.")
        return "cpu"

    while True:
        choice = input("  Your choice [1/2]: ").strip()
        if choice == "1":
            return "cuda"
        if choice == "2":
            return "cpu"
        print("  Please enter 1 or 2.")


def select_models(config: dict) -> list[str]:
    keys = list(config.keys())
    print("  Available architectures:")
    print()
    for index, key in enumerate(keys, start=1):
        cfg = config[key]
        source = cfg.get("source", "torchvision")
        year = cfg.get("year", "-")
        print(f"    [{index:>2}]  {cfg['name']:<22} {source:<12} ({year})")

    print(f"    [{len(keys) + 1:>2}]  -- Run ALL models --")
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


def run_benchmark(model_keys: list[str], device: str) -> None:
    started_at = time.time()

    section("Loading EuroSAT Train / Val / Test Splits")
    step(1, "Downloading or caching EuroSAT and building deterministic splits...")
    _, _, test_loader, split_meta = get_eurosat_loaders(
        input_size=224,
        batch_size=64,
        num_workers=2,
        seed=DEFAULT_SEED,
    )
    split_counts = split_meta["counts"]
    step(
        2,
        "Split ready - "
        f"train={split_counts['train']:,}, val={split_counts['val']:,}, test={split_counts['test']:,}.",
    )
    print("    INFO: Accuracy below is a lower-bound transfer baseline because the 10-class head is untrained.")

    all_results = {}
    config = load_model_config()

    for index, key in enumerate(model_keys, start=1):
        cfg = config[key]
        section(f"Model {index}/{len(model_keys)} - {cfg['name']}")
        print(f"    Paper : {cfg.get('paper', 'N/A')}")
        print(f"    Source: {cfg.get('source', 'torchvision')} | Input: {cfg.get('input_size', 224)}px")
        print()

        try:
            input_size = cfg.get("input_size", 224)

            step(1, "Loading pretrained backbone and resetting a 10-class classifier head...")
            model = get_model(key, num_classes=10, pretrained=True)
            info = get_model_info(model)
            result_line("Parameters", info["param_str"])
            result_line("Disk footprint", f"{info['size_mb']} MB")

            step(2, "Estimating computational cost (MACs / FLOPs)...")
            flops = estimate_flops(model, input_size)
            result_line("MACs", flops.get("macs_str", "N/A"))
            result_line("FLOPs", flops.get("flops_str", "N/A"))

            step(3, "Profiling CPU inference latency (20 runs, 5 warmup)...")
            latency_cpu = measure_latency(model, input_size, "cpu", warmup_runs=5, benchmark_runs=20, batch_size=1)
            result_line("CPU latency (median)", f"{latency_cpu['median_ms']} ms")

            latency_gpu = None
            if device == "cuda":
                step(4, "Profiling GPU inference latency (30 runs, 5 warmup)...")
                latency_gpu = measure_latency(model, input_size, "cuda", warmup_runs=5, benchmark_runs=30, batch_size=1)
                result_line("GPU latency (median)", f"{latency_gpu['median_ms']} ms")

            step(5, "Measuring batch throughput (batch sizes 1, 16)...")
            throughput = measure_throughput(model, input_size, device, batch_sizes=[1, 16], duration_seconds=1.0)
            for batch_size, images_per_second in throughput.items():
                result_line(f"Throughput (bs={batch_size})", f"{images_per_second} img/s")

            step(6, "Evaluating transfer baseline accuracy on the held-out EuroSAT test split...")
            accuracy = evaluate_accuracy(model, test_loader, device=device)
            result_line("Top-1 accuracy", f"{accuracy['top1_accuracy']}%")
            result_line("Top-5 accuracy", f"{accuracy['top5_accuracy']}%")

            print(f"\n    OK: {cfg['name']} complete.")

            all_results[key] = {
                "name": cfg["name"],
                "year": cfg.get("year"),
                "paper": cfg.get("paper"),
                "source": cfg.get("source", "torchvision"),
                "input_size": input_size,
                "protocol": EXPERIMENT_PROTOCOL,
                "model_info": info,
                "flops": flops,
                "latency_cpu": latency_cpu,
                "latency_gpu": latency_gpu,
                "throughput": {str(batch_size): value for batch_size, value in throughput.items()},
                "accuracy": accuracy,
            }
        except Exception as error:
            print(f"\n    ERROR benchmarking {key}: {error}")

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
            "protocol": EXPERIMENT_PROTOCOL,
            "protocol_label": "Head-reset transfer baseline",
            "protocol_description": (
                "Each pretrained backbone is adapted to EuroSAT by replacing the final classifier "
                "with a fresh 10-class layer and evaluating before training."
            ),
            "split_seed": split_meta["seed"],
            "split_counts": split_counts,
            "class_names": split_meta["class_names"],
        },
        "models": all_results,
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    elapsed = time.time() - started_at
    step(1, f"Results saved to {RESULTS_FILE}")
    step(2, f"Total benchmark time: {elapsed / 60:.1f} minutes")
    print()
    print("  ------------------------------------------------")
    print("  Next steps:")
    print("    python 02_generate_visualizations.py")
    print("    python 03_finetune_models.py")
    print("  ------------------------------------------------")
    print()


def main() -> None:
    banner()
    config = load_model_config()
    device = select_device()
    selected_models = select_models(config)
    run_benchmark(selected_models, device)


if __name__ == "__main__":
    main()
