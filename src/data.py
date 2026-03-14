"""
Dataset loading and preprocessing utilities for EuroSAT.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_SEED = 42
DEFAULT_SPLIT_FILE = Path("results/splits/eurosat_split_seed42.json")


class TransformedSubset(Dataset):
    """Apply a transform to a deterministic subset of a base dataset."""

    def __init__(self, dataset: Dataset, indices: Sequence[int], transform=None):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, label = self.dataset[self.indices[index]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_transforms(input_size: int = 224) -> Tuple[T.Compose, T.Compose]:
    """Build training and evaluation transforms for aerial imagery."""
    resize_size = max(int(input_size * 1.14), input_size + 16)

    train_transform = T.Compose([
        T.RandomResizedCrop(input_size, scale=(0.75, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=180),
        T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_transform = T.Compose([
        T.Resize(resize_size),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, eval_transform


def _load_base_dataset(data_dir: str = "./data"):
    return torchvision.datasets.EuroSAT(
        root=data_dir,
        download=True,
        transform=None,
    )


def _extract_targets(dataset) -> List[int]:
    if hasattr(dataset, "targets") and dataset.targets is not None:
        return [int(target) for target in dataset.targets]

    if hasattr(dataset, "samples") and dataset.samples is not None:
        return [int(label) for _, label in dataset.samples]

    return [int(dataset[idx][1]) for idx in range(len(dataset))]


def _build_stratified_split_indices(
    targets: Sequence[int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[int]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    grouped_indices: Dict[int, List[int]] = {}
    for index, label in enumerate(targets):
        grouped_indices.setdefault(int(label), []).append(index)

    rng = random.Random(seed)
    split_indices = {"train": [], "val": [], "test": []}

    for label, label_indices in grouped_indices.items():
        label_indices = list(label_indices)
        rng.shuffle(label_indices)

        n_total = len(label_indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        if n_total >= 3:
            n_train = max(n_train, 1)
            n_val = max(n_val, 1)
            n_test = max(n_test, 1)

            overflow = n_train + n_val + n_test - n_total
            while overflow > 0:
                if n_train >= n_val and n_train > 1:
                    n_train -= 1
                elif n_val > 1:
                    n_val -= 1
                else:
                    n_test -= 1
                overflow -= 1

        split_indices["train"].extend(label_indices[:n_train])
        split_indices["val"].extend(label_indices[n_train:n_train + n_val])
        split_indices["test"].extend(label_indices[n_train + n_val:])

    for split_name in split_indices:
        split_indices[split_name].sort()

    return split_indices


def get_or_create_eurosat_splits(
    data_dir: str = "./data",
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
    split_file: Path | None = None,
    force_recreate: bool = False,
) -> Dict[str, object]:
    """Create or load deterministic stratified EuroSAT split indices."""
    base_dataset = _load_base_dataset(data_dir=data_dir)
    targets = _extract_targets(base_dataset)

    if split_file is None:
        split_file = DEFAULT_SPLIT_FILE if seed == DEFAULT_SEED else Path(
            f"results/splits/eurosat_split_seed{seed}.json"
        )

    split_file = Path(split_file)
    if split_file.exists() and not force_recreate:
        with open(split_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload

    split_indices = _build_stratified_split_indices(targets, train_ratio, val_ratio, seed)
    split_counts = {name: len(indices) for name, indices in split_indices.items()}
    class_counts = {}
    for class_index, class_name in enumerate(getattr(base_dataset, "classes", [])):
        class_counts[class_name] = sum(1 for label in targets if label == class_index)

    payload = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": round(1.0 - train_ratio - val_ratio, 2),
        "counts": split_counts,
        "class_names": list(getattr(base_dataset, "classes", [])),
        "class_counts": class_counts,
        "indices": split_indices,
    }

    split_file.parent.mkdir(parents=True, exist_ok=True)
    with open(split_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return payload


def get_eurosat_datasets(
    input_size: int = 224,
    data_dir: str = "./data",
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
    split_file: Path | None = None,
    force_recreate_split: bool = False,
):
    """Return train/val/test datasets and split metadata for EuroSAT."""
    base_dataset = _load_base_dataset(data_dir=data_dir)
    split_payload = get_or_create_eurosat_splits(
        data_dir=data_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        split_file=split_file,
        force_recreate=force_recreate_split,
    )

    train_transform, eval_transform = get_transforms(input_size=input_size)
    indices = split_payload["indices"]

    train_dataset = TransformedSubset(base_dataset, indices["train"], transform=train_transform)
    val_dataset = TransformedSubset(base_dataset, indices["val"], transform=eval_transform)
    test_dataset = TransformedSubset(base_dataset, indices["test"], transform=eval_transform)

    return train_dataset, val_dataset, test_dataset, split_payload


def get_eurosat_loaders(
    input_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
    data_dir: str = "./data",
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
    split_file: Path | None = None,
    force_recreate_split: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, object]]:
    """Return deterministic EuroSAT train/val/test loaders."""
    train_dataset, val_dataset, test_dataset, split_payload = get_eurosat_datasets(
        input_size=input_size,
        data_dir=data_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        split_file=split_file,
        force_recreate_split=force_recreate_split,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, split_payload


def get_dummy_input(
    batch_size: int = 1,
    input_size: int = 224,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate a random input tensor for latency measurement."""
    return torch.randn(batch_size, 3, input_size, input_size, device=device)
