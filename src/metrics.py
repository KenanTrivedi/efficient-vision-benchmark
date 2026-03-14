"""
Accuracy and model analysis metrics.

Computes Top-1 / Top-5 accuracy on a given dataset and
provides FLOPs estimation using ptflops.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple
from tqdm import tqdm


# ═══════════════════════════════════════════════════
# Classification Accuracy
# ═══════════════════════════════════════════════════

@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    topk: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    """
    Evaluate classification accuracy on a dataset.

    Args:
        model: Trained model
        dataloader: Evaluation data loader
        device: 'cpu' or 'cuda'
        topk: Which top-k accuracies to compute

    Returns:
        Dictionary with top-k accuracy percentages
    """
    model = model.to(device)
    model.eval()

    correct = {k: 0 for k in topk}
    total = 0

    for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        # Compute top-k correct predictions
        maxk = max(topk)
        _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct_mask = pred.eq(labels.view(1, -1).expand_as(pred))

        for k in topk:
            correct[k] += correct_mask[:k].reshape(-1).float().sum().item()

        total += labels.size(0)

    results = {}
    for k in topk:
        acc = 100.0 * correct[k] / total
        results[f"top{k}_accuracy"] = round(acc, 2)

    results["total_samples"] = total
    return results


# ═══════════════════════════════════════════════════
# FLOPs Estimation
# ═══════════════════════════════════════════════════

def estimate_flops(
    model: nn.Module, input_size: int = 224
) -> Dict[str, Any]:
    """
    Estimate FLOPs and MACs for a model using ptflops.

    Args:
        model: The model to profile
        input_size: Spatial resolution of input

    Returns:
        Dictionary with 'flops', 'macs', 'flops_str', 'macs_str'
    """
    try:
        from ptflops import get_model_complexity_info

        macs, params = get_model_complexity_info(
            model,
            (3, input_size, input_size),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )

        return {
            "macs": int(macs),
            "flops": int(macs * 2),  # FLOPs ≈ 2 × MACs
            "macs_str": _format_number(macs),
            "flops_str": _format_number(macs * 2),
        }

    except ImportError:
        print("⚠ ptflops not installed — skipping FLOPs estimation")
        return {"macs": None, "flops": None, "macs_str": "N/A", "flops_str": "N/A"}

    except Exception as e:
        print(f"⚠ FLOPs estimation failed: {e}")
        return {"macs": None, "flops": None, "macs_str": "N/A", "flops_str": "N/A"}


def _format_number(n: float) -> str:
    """Format large numbers with SI prefixes."""
    if n >= 1e9:
        return f"{n / 1e9:.2f}G"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(int(n))
