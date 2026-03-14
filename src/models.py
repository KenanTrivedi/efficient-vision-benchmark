"""
Model loading and preparation utilities.

Loads pretrained torchvision models and adapts classifiers
for the target dataset (e.g., CIFAR-10 with 10 classes).
"""

import torch
import torch.nn as nn
import torchvision.models as models
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


# ═══════════════════════════════════════════════════
# Model Registry
# ═══════════════════════════════════════════════════

def load_model_config(config_path: str = "configs/models.yaml") -> Dict[str, Any]:
    """Load model configurations from YAML registry."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model(
    model_key: str,
    num_classes: int = 10,
    config_path: str = "configs/models.yaml",
) -> nn.Module:
    """
    Load a model from the registry and adapt its classifier head.

    Args:
        model_key: Key from configs/models.yaml (e.g., 'mobilenet_v2')
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        config_path: Path to model config file

    Returns:
        PyTorch model with adapted classifier
    """
    config = load_model_config(config_path)

    if model_key not in config:
        available = ", ".join(config.keys())
        raise ValueError(
            f"Unknown model '{model_key}'. Available: {available}"
        )

    model_cfg = config[model_key]
    tv_name = model_cfg["torchvision_name"]
    weights_name = model_cfg.get("weights", "DEFAULT")

    # Load pretrained model from torchvision
    model = getattr(models, tv_name)(weights=weights_name)

    # Adapt the classifier head for target number of classes
    model = _adapt_classifier(model, tv_name, num_classes)

    return model


def _adapt_classifier(
    model: nn.Module, arch_name: str, num_classes: int
) -> nn.Module:
    """
    Replace the final classification layer to match target classes.

    Different architectures use different attribute names for the
    classifier head — this function handles each case.
    """
    if "resnet" in arch_name or "regnet" in arch_name or "shufflenet" in arch_name:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif "convnext" in arch_name:
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    elif "swin" in arch_name:
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

    elif "maxvit" in arch_name:
        in_features = model.classifier[5].in_features
        model.classifier[5] = nn.Linear(in_features, num_classes)

    elif "mobilenet" in arch_name or "efficientnet" in arch_name:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    else:
        # Fallback: try common attribute names
        if hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(model, "head"):
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes)
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(
                f"Cannot adapt classifier for architecture '{arch_name}'"
            )

    return model


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Compute model statistics: parameter count, size on disk, etc.

    Returns:
        Dictionary with 'total_params', 'trainable_params',
        'size_mb', and 'size_kb'.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    # Estimate model size (float32 = 4 bytes per param)
    size_bytes = total_params * 4
    size_mb = size_bytes / (1024 ** 2)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_mb": round(size_mb, 2),
        "size_kb": round(size_bytes / 1024, 1),
        "param_str": _format_params(total_params),
    }


def _format_params(n: int) -> str:
    """Format parameter count as human-readable string."""
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def list_available_models(config_path: str = "configs/models.yaml") -> None:
    """Print a formatted list of available models."""
    config = load_model_config(config_path)
    print(f"\n{'Model Key':<25} {'Name':<22} {'Description'}")
    print("─" * 80)
    for key, cfg in config.items():
        print(f"{key:<25} {cfg['name']:<22} {cfg.get('description', '')}")
    print()
