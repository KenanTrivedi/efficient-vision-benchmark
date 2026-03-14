"""
Latency and throughput measurement utilities.

Provides reliable timing for single-image inference and
batch throughput across CPU and (optionally) GPU.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
from src.data import get_dummy_input


# ═══════════════════════════════════════════════════
# Timing Utilities
# ═══════════════════════════════════════════════════

@torch.no_grad()
def measure_latency(
    model: nn.Module,
    input_size: int = 224,
    device: str = "cpu",
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    Measure inference latency with proper warmup and synchronization.

    Args:
        model: The model to benchmark
        input_size: Spatial resolution of input
        device: 'cpu' or 'cuda'
        warmup_runs: Number of warmup iterations (not timed)
        benchmark_runs: Number of timed iterations
        batch_size: Number of images per forward pass

    Returns:
        Dictionary with timing statistics (in milliseconds):
        - mean_ms: average latency per batch
        - std_ms: standard deviation
        - median_ms: median latency
        - min_ms: minimum latency
        - max_ms: maximum latency
        - p95_ms: 95th percentile latency
    """
    model = model.to(device)
    model.eval()

    dummy = get_dummy_input(batch_size, input_size, device)

    # Warmup — let CUDA kernels initialize, caches fill, etc.
    for _ in range(warmup_runs):
        _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()

    # Benchmark
    timings = []
    for _ in range(benchmark_runs):
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        _ = model(dummy)

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = (time.perf_counter() - start) * 1000  # ms
        timings.append(elapsed)

    timings = np.array(timings)

    return {
        "mean_ms": round(float(np.mean(timings)), 2),
        "std_ms": round(float(np.std(timings)), 2),
        "median_ms": round(float(np.median(timings)), 2),
        "min_ms": round(float(np.min(timings)), 2),
        "max_ms": round(float(np.max(timings)), 2),
        "p95_ms": round(float(np.percentile(timings, 95)), 2),
        "batch_size": batch_size,
        "device": device,
        "num_runs": benchmark_runs,
    }


@torch.no_grad()
def measure_throughput(
    model: nn.Module,
    input_size: int = 224,
    device: str = "cpu",
    batch_sizes: list = None,
    duration_seconds: float = 5.0,
) -> Dict[int, float]:
    """
    Measure throughput (images/second) at various batch sizes.

    Args:
        model: The model to benchmark
        input_size: Spatial resolution
        device: 'cpu' or 'cuda'
        batch_sizes: List of batch sizes to test
        duration_seconds: How long to run each measurement

    Returns:
        Dictionary mapping batch_size -> images_per_second
    """
    if batch_sizes is None:
        batch_sizes = [1, 16, 64]

    model = model.to(device)
    model.eval()

    results = {}

    for bs in batch_sizes:
        dummy = get_dummy_input(bs, input_size, device)

        # Warmup
        for _ in range(3):
            _ = model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()

        # Measure
        count = 0
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        while time.perf_counter() - start < duration_seconds:
            _ = model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            count += bs

        elapsed = time.perf_counter() - start
        throughput = count / elapsed

        results[bs] = round(throughput, 1)

    return results
