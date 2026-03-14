"""
Training utilities for EuroSAT transfer learning experiments.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List

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


def resolve_device(device: str | None = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_amp_context(device: str):
    if device == "cuda" and hasattr(torch, "amp"):
        return torch.amp.autocast(device_type="cuda")
    from contextlib import nullcontext
    return nullcontext()


def make_grad_scaler(device: str):
    if device == "cuda" and hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return None


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scaler=None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        amp_context = make_amp_context(device)
        with amp_context:
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        predictions = outputs.argmax(dim=1)
        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((predictions == labels).sum().item())
        total_samples += int(labels.size(0))

    return {
        "loss": round(total_loss / total_samples, 4),
        "accuracy": round(100.0 * total_correct / total_samples, 2),
        "samples": total_samples,
    }


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        predictions = outputs.argmax(dim=1)

        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((predictions == labels).sum().item())
        total_samples += int(labels.size(0))

    return {
        "loss": round(total_loss / total_samples, 4),
        "accuracy": round(100.0 * total_correct / total_samples, 2),
        "samples": total_samples,
    }


def save_history_csv(history: List[Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys()) if history else []
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def plot_training_history(history: List[Dict[str, float]], output_path: Path) -> None:
    if not history:
        return

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_acc = [entry["train_accuracy"] for entry in history]
    val_acc = [entry["val_accuracy"] for entry in history]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.patch.set_facecolor("#f7f3ea")

    axes[0].plot(epochs, train_loss, marker="o", color="#0f766e", label="Train loss")
    axes[0].plot(epochs, val_loss, marker="o", color="#b45309", label="Val loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", color="#1d4ed8", label="Train accuracy")
    axes[1].plot(epochs, val_acc, marker="o", color="#dc2626", label="Val accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    for axis in axes:
        axis.set_facecolor("#fffdf8")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
