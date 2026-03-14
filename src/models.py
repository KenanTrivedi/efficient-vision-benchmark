"""
Model loading and preparation utilities.

Loads pretrained models from torchvision or huggingface/timm
and adapts classifiers for the target dataset (e.g., CIFAR-10).
"""

import torch
import torch.nn as nn
import torchvision.models as models
import timm
import yaml
from typing import Dict, Any


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
    Load a model from the registry (torchvision or timm)
    and adapt its classifier head dynamically.
    """
    config = load_model_config(config_path)

    if model_key not in config:
        available = ", ".join(config.keys())
        raise ValueError(f"Unknown model '{model_key}'. Available: {available}")

    model_cfg = config[model_key]
    model_id = model_cfg["model_id"]
    source = model_cfg.get("source", "torchvision")

    # Load from PyTorch Image Models (Hugging Face / timm)
    if source == "timm":
        # timm automatically handles head adaptation if we pass num_classes
        model = timm.create_model(model_id, pretrained=True, num_classes=num_classes)

    # Load from Torchvision
    elif source == "torchvision":
        weights_name = model_cfg.get("weights", "DEFAULT")
        model = getattr(models, model_id)(weights=weights_name)
        model = _adapt_torchvision_classifier(model, model_id, num_classes)
        
    else:
        raise ValueError(f"Unknown model source '{source}'")

    return model


def _adapt_torchvision_classifier(model: nn.Module, arch_name: str, num_classes: int) -> nn.Module:
    """Replace the final classification layer for standard torchvision architectures."""
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
        if hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, "head"):
            model.head = nn.Linear(model.head.in_features, num_classes)
        elif hasattr(model, "classifier"):
            if isinstance(model.classifier, nn.Sequential):
                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
            else:
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Compute parameter count and footprint metadata."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

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
    """Format parameter count as 1.5M, 200K, etc."""
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def list_available_models(config_path: str = "configs/models.yaml") -> None:
    config = load_model_config(config_path)
    print(f"\n{'Model Key':<20} {'Source':<12} {'Name':<18} {'Description'}")
    print("─" * 80)
    for key, cfg in config.items():
        src = cfg.get("source", "torchvision")
        print(f"{key:<20} {src:<12} {cfg['name']:<18} {cfg.get('description', '')[:50]}...")
    print()
