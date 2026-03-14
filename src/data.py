"""
Dataset loading and preprocessing utilities.

Provides standardized data loading for CIFAR-10 with
appropriate transforms for each model architecture.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from typing import Tuple


# ═══════════════════════════════════════════════════
# CIFAR-10 Configuration
# ═══════════════════════════════════════════════════

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(
    input_size: int = 224, use_imagenet_stats: bool = True
) -> Tuple[T.Compose, T.Compose]:
    """
    Build train and evaluation transforms for the aerial dataset.

    Since pretrained models expect ImageNet-normalized 224x224 inputs,
    we explicitly resize the dataset and normalize accordingly.

    Args:
        input_size: Target spatial resolution
        use_imagenet_stats: If True, use ImageNet mean/std (for pretrained
                           models). Otherwise use standard dataset statistics.

    Returns:
        (train_transform, eval_transform)
    """
    mean = IMAGENET_MEAN if use_imagenet_stats else CIFAR10_MEAN
    std = IMAGENET_STD if use_imagenet_stats else CIFAR10_STD

    train_transform = T.Compose([
        T.Resize(input_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),   # Common for aerial
        T.RandomCrop(input_size, padding=4),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    eval_transform = T.Compose([
        T.Resize(input_size),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    return train_transform, eval_transform


def get_eurosat_loaders(
    input_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    data_dir: str = "./data",
) -> Tuple[DataLoader, DataLoader]:
    """
    Load EuroSAT aerial/satellite dataset.
    This simulates edge-inference on drone imagery.
    """
    _, eval_transform = get_transforms(input_size)

    # Note: EuroSAT doesn't have a split default in torchvision,
    # so we load the whole dataset and use a deterministic subset or simply evaluate on it
    dataset = torchvision.datasets.EuroSAT(
        root=data_dir,
        download=True,
        transform=eval_transform,
    )

    # For pure benchmarking speed, we just use a dataloader of the dataset
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader, loader


def get_dummy_input(
    batch_size: int = 1, input_size: int = 224, device: str = "cpu"
) -> torch.Tensor:
    """Generate a random input tensor for latency measurement."""
    return torch.randn(batch_size, 3, input_size, input_size, device=device)
