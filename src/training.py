"""
Training utilities for EuroSAT transfer-learning experiments.
"""

from __future__ import annotations

import csv
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str | None = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_amp_context(device: str):
    if device == "cuda" and hasattr(torch, "amp"):
        return torch.amp.autocast(device_type="cuda")
    return nullcontext()


def make_grad_scaler(device: str):
    if device == "cuda" and hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return None


def cosine_schedule_factor(
    epoch_index: int,
    total_epochs: int,
    warmup_epochs: int = 1,
    min_lr_scale: float = 0.05,
) -> float:
    if total_epochs <= 1:
        return 1.0

    warmup_epochs = max(0, min(int(warmup_epochs), total_epochs - 1))
    if warmup_epochs > 0 and epoch_index < warmup_epochs:
        progress = (epoch_index + 1) / warmup_epochs
        return 0.2 + 0.8 * progress

    cosine_steps = max(total_epochs - warmup_epochs - 1, 1)
    progress = (epoch_index - warmup_epochs) / cosine_steps
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_scale + (1.0 - min_lr_scale) * cosine


def apply_epoch_lr_schedule(
    optimizer: torch.optim.Optimizer,
    base_lrs: Sequence[float],
    *,
    epoch_index: int,
    total_epochs: int,
    warmup_epochs: int = 1,
    min_lr_scale: float = 0.05,
) -> Dict[str, float]:
    scale = cosine_schedule_factor(
        epoch_index=epoch_index,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        min_lr_scale=min_lr_scale,
    )
    current_lrs: Dict[str, float] = {}
    for index, (param_group, base_lr) in enumerate(zip(optimizer.param_groups, base_lrs)):
        param_group["lr"] = float(base_lr) * scale
        group_name = param_group.get("name", f"group_{index}")
        current_lrs[group_name] = round(float(param_group["lr"]), 8)
    return current_lrs


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    *,
    scaler=None,
    grad_clip_norm: float | None = None,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, (images, labels) in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with make_amp_context(device):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

        predictions = outputs.argmax(dim=1)
        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((predictions == labels).sum().item())
        total_samples += int(labels.size(0))

    return {
        "loss": round(total_loss / max(total_samples, 1), 4),
        "accuracy": round(100.0 * total_correct / max(total_samples, 1), 2),
        "samples": total_samples,
    }


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    *,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, (images, labels) in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        predictions = outputs.argmax(dim=1)

        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((predictions == labels).sum().item())
        total_samples += int(labels.size(0))

    return {
        "loss": round(total_loss / max(total_samples, 1), 4),
        "accuracy": round(100.0 * total_correct / max(total_samples, 1), 2),
        "samples": total_samples,
    }


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    *,
    max_batches: int | None = None,
) -> Dict[str, np.ndarray]:
    model.eval()
    all_targets: List[np.ndarray] = []
    all_predictions: List[np.ndarray] = []

    for batch_index, (images, labels) in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        outputs = model(images.to(device, non_blocking=True))
        predictions = outputs.argmax(dim=1).cpu().numpy()

        all_targets.append(labels.numpy())
        all_predictions.append(predictions)

    targets = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.int64)
    predictions = np.concatenate(all_predictions) if all_predictions else np.array([], dtype=np.int64)
    return {
        "targets": targets,
        "predictions": predictions,
    }


def build_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        confusion[int(target), int(prediction)] += 1
    return confusion


def per_class_accuracy_from_confusion(
    confusion: np.ndarray,
    class_names: Sequence[str],
) -> List[Dict[str, float | int | str]]:
    rows = []
    for class_index, class_name in enumerate(class_names):
        support = int(confusion[class_index].sum())
        correct = int(confusion[class_index, class_index])
        accuracy = 100.0 * correct / support if support else 0.0
        rows.append({
            "class_name": class_name,
            "support": support,
            "correct": correct,
            "accuracy": round(accuracy, 2),
        })
    return rows


def save_history_csv(history: List[Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys()) if history else []
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _mark_stage_transitions(axes, history: List[Dict[str, float]]) -> None:
    for index in range(1, len(history)):
        if history[index].get("stage") == history[index - 1].get("stage"):
            continue
        epoch_value = history[index]["epoch"] - 0.5
        for axis in axes:
            axis.axvline(epoch_value, color="#94a3b8", linestyle="--", linewidth=1.1, alpha=0.9)


def plot_training_history(history: List[Dict[str, float]], output_path: Path) -> None:
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_acc = [entry["train_accuracy"] for entry in history]
    val_acc = [entry["val_accuracy"] for entry in history]
    lr_backbone = [entry.get("lr_backbone", 0.0) for entry in history]
    lr_head = [entry.get("lr_head", entry.get("lr_backbone", 0.0)) for entry in history]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8))
    fig.patch.set_facecolor("#f7f3ea")

    axes[0].plot(epochs, train_loss, marker="o", color="#0f766e", label="Train loss")
    axes[0].plot(epochs, val_loss, marker="o", color="#b45309", label="Val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, train_acc, marker="o", color="#1d4ed8", label="Train accuracy")
    axes[1].plot(epochs, val_acc, marker="o", color="#dc2626", label="Val accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(frameon=False)

    axes[2].plot(epochs, lr_backbone, marker="o", color="#334155", label="Backbone LR")
    axes[2].plot(epochs, lr_head, marker="o", color="#7c3aed", label="Head LR")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_yscale("log")
    axes[2].legend(frameon=False)

    _mark_stage_transitions(axes, history)

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_facecolor("#fffdf8")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    confusion: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    if confusion.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8.5, 7.4))
    fig.patch.set_facecolor("#f7f3ea")
    image = ax.imshow(confusion, cmap="YlGnBu")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Test-set Confusion Matrix")

    threshold = confusion.max() * 0.6 if confusion.size else 0
    for row_index in range(confusion.shape[0]):
        for col_index in range(confusion.shape[1]):
            value = int(confusion[row_index, col_index])
            color = "white" if value > threshold else "#18212b"
            ax.text(col_index, row_index, value, ha="center", va="center", color=color, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
