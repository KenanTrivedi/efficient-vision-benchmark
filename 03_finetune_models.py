#!/usr/bin/env python3
"""
03_finetune_models.py - EuroSAT transfer-learning pipeline
===========================================================
Fine-tunes the recommended models on deterministic EuroSAT splits and
stores checkpoints, learning curves, and a summary JSON for comparison
against the transfer baseline benchmark.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.data import DEFAULT_SEED, get_eurosat_loaders
from src.metrics import evaluate_accuracy
from src.models import (
    get_model,
    get_model_info,
    load_model_config,
    set_finetune_strategy,
)
from src.training import (
    evaluate_classifier,
    make_grad_scaler,
    plot_training_history,
    resolve_device,
    save_history_csv,
    seed_everything,
    train_one_epoch,
)


RECOMMENDED_MODELS = [
    "mobilevitv2_050",
    "mobilenet_v3_large",
    "convnext_tiny",
]
RESULTS_DIR = Path("results/finetune")
SUMMARY_FILE = RESULTS_DIR / "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune EuroSAT models on deterministic splits.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=RECOMMENDED_MODELS,
        help="Model keys to fine-tune. Defaults to the recommended trio.",
    )
    parser.add_argument(
        "--strategy",
        default="head",
        choices=["head", "full"],
        help="Fine-tuning strategy: only train the classifier head or the full network.",
    )
    parser.add_argument("--epochs", type=int, default=8, help="Maximum number of epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=128, help="Validation/test batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience on validation accuracy.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--device", default=None, help="Force training on a specific device.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for splits and training.")
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit batches per epoch for smoke tests.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=None,
        help="Limit validation/test batches for smoke tests.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available model keys and exit.",
    )
    return parser.parse_args()


def _limited_loader_batches(loader, max_batches: int | None):
    if max_batches is None:
        yield from loader
        return

    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        yield batch


def train_one_epoch_limited(model, loader, optimizer, criterion, device, scaler=None, max_batches=None):
    if max_batches is None:
        return train_one_epoch(model, loader, optimizer, criterion, device, scaler=scaler)

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in _limited_loader_batches(loader, max_batches):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and device == "cuda":
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
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
def evaluate_classifier_limited(model, loader, criterion, device, max_batches=None):
    if max_batches is None:
        return evaluate_classifier(model, loader, criterion, device)

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in _limited_loader_batches(loader, max_batches):
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


@torch.no_grad()
def evaluate_accuracy_limited(model, loader, device, max_batches=None):
    if max_batches is None:
        return evaluate_accuracy(model, loader, device=device)

    model = model.to(device)
    model.eval()

    correct_top1 = 0.0
    correct_top5 = 0.0
    total = 0

    for images, labels in _limited_loader_batches(loader, max_batches):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        _, pred = outputs.topk(5, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct_mask = pred.eq(labels.view(1, -1).expand_as(pred))
        correct_top1 += correct_mask[:1].reshape(-1).float().sum().item()
        correct_top5 += correct_mask[:5].reshape(-1).float().sum().item()
        total += labels.size(0)

    return {
        "top1_accuracy": round(100.0 * correct_top1 / total, 2),
        "top5_accuracy": round(100.0 * correct_top5 / total, 2),
        "total_samples": total,
    }


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)


def run_for_model(model_key: str, config: dict, args: argparse.Namespace, device: str, split_meta: dict) -> dict:
    started_at = time.time()
    model_name = config[model_key]["name"]
    run_dir = RESULTS_DIR / model_key
    run_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(f"  --- Fine-tuning {model_name} " + "-" * max(0, 34 - len(model_name)))

    train_loader, val_loader, test_loader, _ = get_eurosat_loaders(
        input_size=config[model_key].get("input_size", 224),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    _, val_loader_eval, test_loader_eval, _ = get_eurosat_loaders(
        input_size=config[model_key].get("input_size", 224),
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = get_model(model_key, num_classes=10, pretrained=True)
    model = set_finetune_strategy(model, args.strategy)
    model = model.to(device)

    model_info = get_model_info(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = make_grad_scaler(device)

    best_val_accuracy = -1.0
    best_epoch = 0
    best_checkpoint = run_dir / "best.pt"
    history = []
    epochs_without_improvement = 0

    print(f"    Strategy           : {args.strategy}")
    print(f"    Trainable params   : {model_info['trainable_param_str']}")
    print(f"    Batch size         : {args.batch_size}")
    print(f"    Max epochs         : {args.epochs}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch_limited(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            max_batches=args.max_train_batches,
        )
        val_metrics = evaluate_classifier_limited(
            model,
            val_loader_eval,
            criterion,
            device,
            max_batches=args.max_eval_batches,
        )
        scheduler.step()

        history_entry = {
            "epoch": epoch,
            "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(history_entry)

        print(
            f"    Epoch {epoch:02d} | "
            f"train acc {train_metrics['accuracy']:>6.2f}% | "
            f"val acc {val_metrics['accuracy']:>6.2f}% | "
            f"val loss {val_metrics['loss']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_key": model_key,
                    "model_name": model_name,
                    "strategy": args.strategy,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_accuracy": best_val_accuracy,
                    "seed": args.seed,
                },
                best_checkpoint,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"    Early stopping triggered after {args.patience} stale epochs.")
            break

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    test_metrics = evaluate_accuracy_limited(
        model,
        test_loader_eval,
        device=device,
        max_batches=args.max_eval_batches,
    )

    history_csv = run_dir / "history.csv"
    history_plot = run_dir / "training_curves.png"
    metrics_file = run_dir / "metrics.json"

    save_history_csv(history, history_csv)
    plot_training_history(history, history_plot)

    baseline_accuracy = None
    baseline_results_file = Path("results/benchmark_results.json")
    if baseline_results_file.exists():
        with open(baseline_results_file, "r", encoding="utf-8") as handle:
            baseline_payload = json.load(handle)
        baseline_accuracy = baseline_payload.get("models", {}).get(model_key, {}).get("accuracy", {}).get("top1_accuracy")

    result_payload = {
        "model_key": model_key,
        "name": model_name,
        "strategy": args.strategy,
        "seed": args.seed,
        "split_seed": split_meta["seed"],
        "epochs_requested": args.epochs,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "model_info": model_info,
        "best_val_accuracy": best_val_accuracy,
        "baseline_top1_accuracy": baseline_accuracy,
        "test_metrics": test_metrics,
        "artifacts": {
            "checkpoint": str(best_checkpoint.as_posix()),
            "history_csv": str(history_csv.as_posix()),
            "training_curves": str(history_plot.as_posix()),
        },
        "elapsed_minutes": round((time.time() - started_at) / 60.0, 2),
    }

    if baseline_accuracy is not None:
        result_payload["accuracy_gain"] = round(test_metrics["top1_accuracy"] - baseline_accuracy, 2)

    with open(metrics_file, "w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2)

    print(f"    Best epoch         : {best_epoch}")
    print(f"    Test accuracy      : {test_metrics['top1_accuracy']}% top-1")
    if baseline_accuracy is not None:
        print(f"    Accuracy lift      : +{result_payload['accuracy_gain']} pts vs. baseline")

    return result_payload


def write_summary(results: dict, args: argparse.Namespace, split_meta: dict, device: str) -> None:
    accuracy_comparison = []
    for model_key, payload in results.items():
        if payload.get("baseline_top1_accuracy") is None:
            continue
        accuracy_comparison.append({
            "model_key": model_key,
            "name": payload["name"],
            "baseline_top1_accuracy": payload["baseline_top1_accuracy"],
            "finetuned_top1_accuracy": payload["test_metrics"]["top1_accuracy"],
            "gain": payload.get("accuracy_gain"),
        })

    summary_payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "device": device,
            "strategy": args.strategy,
            "epochs": args.epochs,
            "seed": args.seed,
            "split_counts": split_meta["counts"],
        },
        "models": results,
        "accuracy_comparison": accuracy_comparison,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    config = load_model_config()

    if args.list_models:
        for model_key, model_cfg in config.items():
            print(f"{model_key:<22} {model_cfg['name']}")
        return

    missing_models = [model_key for model_key in args.models if model_key not in config]
    if missing_models:
        raise SystemExit(f"Unknown model keys: {', '.join(missing_models)}")

    device = resolve_device(args.device)
    seed_everything(args.seed)

    print()
    print("  ==========================================================")
    print("    EUROSAT FINE-TUNING PIPELINE")
    print("    Linear-probe / full-finetune transfer learning")
    print("  ==========================================================")
    print()
    print(f"  Device      : {device}")
    print(f"  Strategy    : {args.strategy}")
    print(f"  Models      : {', '.join(args.models)}")

    _, _, _, split_meta = get_eurosat_loaders(
        input_size=224,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    counts = split_meta["counts"]
    print(f"  Split       : train={counts['train']}, val={counts['val']}, test={counts['test']}")
    print()

    results = {}
    for model_key in args.models:
        results[model_key] = run_for_model(model_key, config, args, device, split_meta)

    write_summary(results, args, split_meta, device)

    print()
    print("  ------------------------------------------------")
    print(f"  Summary saved to {SUMMARY_FILE}")
    print("  Re-run python 02_generate_visualizations.py to refresh the dashboard.")
    print("  ------------------------------------------------")
    print()


if __name__ == "__main__":
    main()
