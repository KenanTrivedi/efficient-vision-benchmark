"""
Model loading and preparation utilities.
"""

from __future__ import annotations

from typing import Any, Dict

import timm
import torch
import torch.nn as nn
import torchvision.models as models
import yaml


def load_model_config(config_path: str = "configs/models.yaml") -> Dict[str, Any]:
    """Load model configurations from the YAML registry."""
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_model(
    model_key: str,
    num_classes: int = 10,
    config_path: str = "configs/models.yaml",
    pretrained: bool = True,
) -> nn.Module:
    """Load a model from the registry and adapt its classifier head."""
    config = load_model_config(config_path)
    if model_key not in config:
        available = ", ".join(config.keys())
        raise ValueError(f"Unknown model '{model_key}'. Available: {available}")

    model_cfg = config[model_key]
    model_id = model_cfg["model_id"]
    source = model_cfg.get("source", "torchvision")

    if source == "timm":
        model = timm.create_model(model_id, pretrained=pretrained, num_classes=num_classes)
        model = _adapt_timm_classifier(model, num_classes)
    elif source == "torchvision":
        weights_name = model_cfg.get("weights", "DEFAULT") if pretrained else None
        model = getattr(models, model_id)(weights=weights_name)
        model = _adapt_torchvision_classifier(model, model_id, num_classes)
    else:
        raise ValueError(f"Unknown model source '{source}'")

    return model


def _adapt_timm_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    """Reset the classifier for timm models when supported."""
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes=num_classes)
    return model


def _adapt_torchvision_classifier(model: nn.Module, arch_name: str, num_classes: int) -> nn.Module:
    """Replace the final classification layer for torchvision architectures."""
    if "resnet" in arch_name or "regnet" in arch_name or "shufflenet" in arch_name:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif "convnext" in arch_name:
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif "swin" in arch_name:
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif "maxvit" in arch_name:
        model.classifier[5] = nn.Linear(model.classifier[5].in_features, num_classes)
    elif "mobilenet" in arch_name or "efficientnet" in arch_name:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
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
        else:
            raise ValueError(f"Do not know how to adapt classifier for '{arch_name}'")
    return model


def get_classifier_module(model: nn.Module) -> nn.Module:
    """Return the terminal classifier module for a supported architecture."""
    if hasattr(model, "get_classifier"):
        classifier = model.get_classifier()
        if isinstance(classifier, nn.Module):
            return classifier

    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        return model.fc
    if hasattr(model, "head") and isinstance(model.head, nn.Module):
        return model.head
    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Sequential):
            return classifier[-1]
        if isinstance(classifier, nn.Module):
            return classifier

    raise ValueError("Unable to locate classifier module for this model")


def freeze_for_linear_probe(model: nn.Module) -> nn.Module:
    """Freeze the backbone and leave only the classifier trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    classifier = get_classifier_module(model)
    for parameter in classifier.parameters():
        parameter.requires_grad = True

    return model


def set_finetune_strategy(model: nn.Module, strategy: str) -> nn.Module:
    """Apply a trainable-parameter strategy to the model."""
    strategy = strategy.lower()
    if strategy in {"head", "linear_probe"}:
        return freeze_for_linear_probe(model)
    if strategy in {"full", "full_finetune"}:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return model
    raise ValueError(f"Unknown finetune strategy '{strategy}'")


def get_trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """Compute parameter count and footprint metadata."""
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    size_bytes = total_params * 4
    size_mb = size_bytes / (1024 ** 2)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "size_mb": round(size_mb, 2),
        "size_kb": round(size_bytes / 1024, 1),
        "param_str": _format_params(total_params),
        "trainable_param_str": _format_params(trainable_params),
    }


def _format_params(value: int) -> str:
    """Format a parameter count as 1.5M, 200K, etc."""
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}K"
    return str(value)


def list_available_models(config_path: str = "configs/models.yaml") -> None:
    config = load_model_config(config_path)
    print(f"\n{'Model Key':<22} {'Source':<12} {'Name':<20} {'Description'}")
    print("-" * 92)
    for key, cfg in config.items():
        src = cfg.get("source", "torchvision")
        description = cfg.get("description", "")
        print(f"{key:<22} {src:<12} {cfg['name']:<20} {description[:48]}...")
    print()
