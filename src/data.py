"""
Dataset loading and preprocessing utilities for EuroSAT.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_SEED = 42
DEFAULT_SPLIT_FILE = Path("results/splits/eurosat_split_seed42.json")
EXPECTED_EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


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
        T.RandomResizedCrop(input_size, scale=(0.72, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=180),
        T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.1, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
    ])

    eval_transform = T.Compose([
        T.Resize(resize_size),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, eval_transform


def inspect_local_eurosat_root(data_dir: str = "./data") -> Dict[str, object]:
    """Inspect the local EuroSAT folder before loading it with torchvision."""
    data_root = Path(data_dir) / "eurosat" / "2750"
    if not data_root.exists():
        return {
            "exists": False,
            "root": str(data_root.as_posix()),
            "class_names": [],
            "missing_classes": EXPECTED_EUROSAT_CLASSES,
            "unexpected_classes": [],
        }

    class_names = sorted(path.name for path in data_root.iterdir() if path.is_dir())
    expected = set(EXPECTED_EUROSAT_CLASSES)
    actual = set(class_names)
    return {
        "exists": True,
        "root": str(data_root.as_posix()),
        "class_names": class_names,
        "missing_classes": sorted(expected - actual),
        "unexpected_classes": sorted(actual - expected),
    }


def _load_base_dataset(data_dir: str = "./data"):
    dataset = torchvision.datasets.EuroSAT(
        root=data_dir,
        download=True,
        transform=None,
    )
    _validate_dataset_integrity(dataset, data_dir=data_dir)
    return dataset


def _validate_dataset_integrity(dataset, data_dir: str) -> None:
    class_names = list(getattr(dataset, "classes", []))
    if class_names == EXPECTED_EUROSAT_CLASSES:
        return

    expected = set(EXPECTED_EUROSAT_CLASSES)
    actual = set(class_names)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    inspection = inspect_local_eurosat_root(data_dir=data_dir)
    raise RuntimeError(
        "The local EuroSAT cache is not the canonical 10-class RGB dataset. "
        f"Found classes={class_names}. Missing={missing or 'none'}. Unexpected={unexpected or 'none'}. "
        f"Inspected folder: {inspection['root']}. Replace the cached data with a clean torchvision download "
        "before running benchmarks or fine-tuning."
    )


def _extract_targets(dataset) -> List[int]:
    if hasattr(dataset, "targets") and dataset.targets is not None:
        return [int(target) for target in dataset.targets]

    if hasattr(dataset, "samples") and dataset.samples is not None:
        return [int(label) for _, label in dataset.samples]

    return [int(dataset[idx][1]) for idx in range(len(dataset))]


def _summarize_class_counts(class_names: Sequence[str], targets: Sequence[int]) -> Dict[str, int]:
    class_counts = {class_name: 0 for class_name in class_names}
    for label in targets:
        class_counts[class_names[int(label)]] += 1
    return class_counts


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

    for label_indices in grouped_indices.values():
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


def _split_payload_matches_dataset(
    payload: Dict[str, object],
    *,
    class_names: Sequence[str],
    class_counts: Dict[str, int],
    total_samples: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> bool:
    required_keys = {"seed", "train_ratio", "val_ratio", "class_names", "class_counts", "indices"}
    if not required_keys.issubset(payload.keys()):
        return False

    if int(payload.get("seed", -1)) != int(seed):
        return False
    if float(payload.get("train_ratio", -1.0)) != float(train_ratio):
        return False
    if float(payload.get("val_ratio", -1.0)) != float(val_ratio):
        return False
    if list(payload.get("class_names", [])) != list(class_names):
        return False
    if payload.get("class_counts", {}) != class_counts:
        return False
    if int(payload.get("total_samples", -1)) != int(total_samples):
        return False

    indices = payload.get("indices", {})
    split_keys = {"train", "val", "test"}
    if set(indices.keys()) != split_keys:
        return False

    observed_indices = []
    for split_name in ("train", "val", "test"):
        split_indices = list(indices.get(split_name, []))
        observed_indices.extend(split_indices)

    return len(observed_indices) == total_samples and sorted(observed_indices) == list(range(total_samples))


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
    class_names = list(getattr(base_dataset, "classes", []))
    class_counts = _summarize_class_counts(class_names, targets)
    total_samples = len(targets)

    if split_file is None:
        split_file = DEFAULT_SPLIT_FILE if seed == DEFAULT_SEED else Path(
            f"results/splits/eurosat_split_seed{seed}.json"
        )

    split_file = Path(split_file)
    if split_file.exists() and not force_recreate:
        with open(split_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if _split_payload_matches_dataset(
            payload,
            class_names=class_names,
            class_counts=class_counts,
            total_samples=total_samples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
        ):
            return payload

    split_indices = _build_stratified_split_indices(targets, train_ratio, val_ratio, seed)
    split_counts = {name: len(indices) for name, indices in split_indices.items()}

    payload = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": round(1.0 - train_ratio - val_ratio, 2),
        "total_samples": total_samples,
        "counts": split_counts,
        "class_names": class_names,
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
