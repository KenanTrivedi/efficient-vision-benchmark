#!/usr/bin/env python3
"""
03_finetune_models.py - EuroSAT transfer-learning pipeline
===========================================================
Runs a staged EuroSAT adaptation recipe for the recommended backbones:

1. Linear-probe warmup on the new classifier head
2. Full-network fine-tuning with a lower backbone learning rate

Outputs checkpoints, CSV history, confusion matrices, and a summary JSON
that feeds the dashboard and deployment-export stages.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.data import DEFAULT_SEED, get_eurosat_loaders
from src.metrics import evaluate_accuracy
from src.models import (
    get_model,
    get_model_info,
    get_optimizer_parameter_groups,
    load_model_config,
    set_finetune_strategy,
)
from src.training import (
    apply_epoch_lr_schedule,
    build_confusion_matrix,
    collect_predictions,
    evaluate_classifier,
    make_grad_scaler,
    per_class_accuracy_from_confusion,
    plot_confusion_matrix,
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
DEFAULT_RECIPE_FILE = Path("configs/finetune_recipes.yaml")
BASELINE_RESULTS_FILE = Path("results/benchmark_results.json")


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
        default=None,
        choices=["staged", "head", "full"],
        help="Override the recipe strategy: staged, head-only, or full-network.",
    )
    parser.add_argument(
        "--recipe-file",
        default=str(DEFAULT_RECIPE_FILE),
        help="YAML file with default and per-model fine-tuning recipes.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Override validation/test batch size.")
    parser.add_argument("--head-lr", type=float, default=None, help="Classifier learning rate.")
    parser.add_argument("--backbone-lr", type=float, default=None, help="Backbone learning rate for full fine-tuning.")
    parser.add_argument("--weight-decay", type=float, default=None, help="AdamW weight decay.")
    parser.add_argument("--label-smoothing", type=float, default=None, help="Cross-entropy label smoothing.")
    parser.add_argument("--linear-probe-epochs", type=int, default=None, help="Warmup epochs with a frozen backbone.")
    parser.add_argument("--finetune-epochs", type=int, default=None, help="Full-network fine-tuning epochs.")
    parser.add_argument("--warmup-epochs", type=int, default=None, help="Epochs used for LR warmup inside each stage.")
    parser.add_argument("--min-lr-scale", type=float, default=None, help="Final cosine LR scale factor.")
    parser.add_argument("--grad-clip-norm", type=float, default=None, help="Gradient clipping norm.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience on validation accuracy.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--device", default=None, help="Force training on a specific device.")
    parser.add_argument("--data-dir", default="./data", help="EuroSAT root directory.")
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


def load_recipe_book(recipe_file: str) -> dict:
    with open(recipe_file, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_recipe(model_key: str, args: argparse.Namespace, recipe_book: dict) -> dict:
    defaults = dict(recipe_book.get("defaults", {}))
    defaults.update(recipe_book.get("models", {}).get(model_key, {}))

    overrides = {
        "strategy": args.strategy,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "head_lr": args.head_lr,
        "backbone_lr": args.backbone_lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "linear_probe_epochs": args.linear_probe_epochs,
        "finetune_epochs": args.finetune_epochs,
        "warmup_epochs": args.warmup_epochs,
        "min_lr_scale": args.min_lr_scale,
        "grad_clip_norm": args.grad_clip_norm,
        "patience": args.patience,
        "num_workers": args.num_workers,
    }
    for key, value in overrides.items():
        if value is not None:
            defaults[key] = value

    defaults.setdefault("strategy", "staged")
    defaults.setdefault("batch_size", 64)
    defaults.setdefault("eval_batch_size", 128)
    defaults.setdefault("head_lr", 2e-3)
    defaults.setdefault("backbone_lr", 2e-4)
    defaults.setdefault("weight_decay", 1e-4)
    defaults.setdefault("label_smoothing", 0.1)
    defaults.setdefault("linear_probe_epochs", 3)
    defaults.setdefault("finetune_epochs", 12)
    defaults.setdefault("warmup_epochs", 1)
    defaults.setdefault("min_lr_scale", 0.05)
    defaults.setdefault("grad_clip_norm", 1.0)
    defaults.setdefault("patience", 4)
    defaults.setdefault("num_workers", 4)

    return defaults


def build_stage_plan(strategy: str, recipe: dict) -> list[dict]:
    if strategy == "staged":
        plan = []
        if recipe["linear_probe_epochs"] > 0:
            plan.append({
                "name": "linear_probe",
                "epochs": recipe["linear_probe_epochs"],
                "finetune_strategy": "head",
            })
        plan.append({
            "name": "full_finetune",
            "epochs": recipe["finetune_epochs"],
            "finetune_strategy": "full",
        })
        return plan

    if strategy == "head":
        epochs = recipe["linear_probe_epochs"] or recipe["finetune_epochs"]
        return [{
            "name": "linear_probe",
            "epochs": epochs,
            "finetune_strategy": "head",
        }]

    epochs = recipe["finetune_epochs"] or recipe["linear_probe_epochs"]
    return [{
        "name": "full_finetune",
        "epochs": epochs,
        "finetune_strategy": "full",
    }]


def build_optimizer(model: nn.Module, stage: dict, recipe: dict) -> tuple[torch.optim.Optimizer, list[float]]:
    model = set_finetune_strategy(model, stage["finetune_strategy"])

    if stage["finetune_strategy"] == "head":
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            [{"params": parameters, "lr": recipe["head_lr"], "name": "head"}],
            weight_decay=recipe["weight_decay"],
        )
        return optimizer, [recipe["head_lr"]]

    parameter_groups = get_optimizer_parameter_groups(model)
    optimizer_groups = []
    base_lrs = []
    if parameter_groups["backbone"]:
        optimizer_groups.append({
            "params": parameter_groups["backbone"],
            "lr": recipe["backbone_lr"],
            "name": "backbone",
        })
        base_lrs.append(recipe["backbone_lr"])
    optimizer_groups.append({
        "params": parameter_groups["head"],
        "lr": recipe["head_lr"],
        "name": "head",
    })
    base_lrs.append(recipe["head_lr"])

    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=recipe["weight_decay"])
    return optimizer, base_lrs


def load_baseline_accuracy(model_key: str) -> float | None:
    if not BASELINE_RESULTS_FILE.exists():
        return None

    with open(BASELINE_RESULTS_FILE, "r", encoding="utf-8") as handle:
        baseline_payload = json.load(handle)
    return baseline_payload.get("models", {}).get(model_key, {}).get("accuracy", {}).get("top1_accuracy")


def evaluate_topk_accuracy(
    model: nn.Module,
    loader,
    device: str,
    *,
    max_batches: int | None = None,
) -> dict:
    if max_batches is None:
        return evaluate_accuracy(model, loader, device=device)

    model.eval()
    correct_top1 = 0.0
    correct_top5 = 0.0
    total = 0

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            if batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            _, pred = outputs.topk(min(5, outputs.shape[1]), dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct_mask = pred.eq(labels.view(1, -1).expand_as(pred))
            correct_top1 += correct_mask[:1].reshape(-1).float().sum().item()
            correct_top5 += correct_mask[: min(5, outputs.shape[1])].reshape(-1).float().sum().item()
            total += labels.size(0)

    return {
        "top1_accuracy": round(100.0 * correct_top1 / max(total, 1), 2),
        "top5_accuracy": round(100.0 * correct_top5 / max(total, 1), 2),
        "total_samples": total,
    }


def run_for_model(
    model_key: str,
    model_config: dict,
    args: argparse.Namespace,
    device: str,
    split_meta: dict,
    recipe: dict,
) -> dict:
    started_at = time.time()
    model_name = model_config[model_key]["name"]
    input_size = model_config[model_key].get("input_size", 224)
    num_classes = len(split_meta["class_names"])

    run_dir = RESULTS_DIR / model_key
    run_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(f"  --- Fine-tuning {model_name} " + "-" * max(0, 34 - len(model_name)))

    train_loader, _, _, _ = get_eurosat_loaders(
        input_size=input_size,
        batch_size=recipe["batch_size"],
        num_workers=recipe["num_workers"],
        data_dir=args.data_dir,
        seed=args.seed,
    )
    _, val_loader, test_loader, _ = get_eurosat_loaders(
        input_size=input_size,
        batch_size=recipe["eval_batch_size"],
        num_workers=recipe["num_workers"],
        data_dir=args.data_dir,
        seed=args.seed,
    )

    model = get_model(model_key, num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=recipe["label_smoothing"])
    scaler = make_grad_scaler(device)
    model_info = get_model_info(model)
    stage_plan = build_stage_plan(recipe["strategy"], recipe)

    best_val_accuracy = -1.0
    best_epoch = 0
    best_stage = None
    best_checkpoint = run_dir / "best.pt"
    history = []
    epochs_without_improvement = 0
    global_epoch = 0

    recipe_file = run_dir / "resolved_recipe.json"
    with open(recipe_file, "w", encoding="utf-8") as handle:
        json.dump(recipe, handle, indent=2)

    print(f"    Strategy           : {recipe['strategy']}")
    print(f"    Input size         : {input_size}")
    print(f"    Total params       : {model_info['param_str']}")
    print(f"    Batch size         : {recipe['batch_size']}")
    print(f"    Eval batch size    : {recipe['eval_batch_size']}")

    for stage in stage_plan:
        optimizer, base_lrs = build_optimizer(model, stage, recipe)
        stage_info = get_model_info(model)

        print(f"    Stage              : {stage['name']} ({stage['epochs']} epochs)")
        print(f"      Trainable params : {stage_info['trainable_param_str']}")

        for stage_epoch in range(1, stage["epochs"] + 1):
            global_epoch += 1
            current_lrs = apply_epoch_lr_schedule(
                optimizer,
                base_lrs,
                epoch_index=stage_epoch - 1,
                total_epochs=stage["epochs"],
                warmup_epochs=recipe["warmup_epochs"],
                min_lr_scale=recipe["min_lr_scale"],
            )

            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler=scaler,
                grad_clip_norm=recipe["grad_clip_norm"],
                max_batches=args.max_train_batches,
            )
            val_metrics = evaluate_classifier(
                model,
                val_loader,
                criterion,
                device,
                max_batches=args.max_eval_batches,
            )

            history_entry = {
                "epoch": global_epoch,
                "stage": stage["name"],
                "stage_epoch": stage_epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "lr_backbone": current_lrs.get("backbone", 0.0),
                "lr_head": current_lrs.get("head", current_lrs.get("backbone", 0.0)),
            }
            history.append(history_entry)

            print(
                f"      Epoch {global_epoch:02d} | "
                f"stage {stage_epoch:02d}/{stage['epochs']:02d} | "
                f"train {train_metrics['accuracy']:>6.2f}% | "
                f"val {val_metrics['accuracy']:>6.2f}% | "
                f"val loss {val_metrics['loss']:.4f}"
            )

            if val_metrics["accuracy"] > best_val_accuracy:
                best_val_accuracy = val_metrics["accuracy"]
                best_epoch = global_epoch
                best_stage = stage["name"]
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_key": model_key,
                        "model_name": model_name,
                        "epoch": global_epoch,
                        "stage": stage["name"],
                        "state_dict": model.state_dict(),
                        "val_accuracy": best_val_accuracy,
                        "seed": args.seed,
                        "recipe": recipe,
                        "class_names": split_meta["class_names"],
                        "input_size": input_size,
                    },
                    best_checkpoint,
                )
            else:
                epochs_without_improvement += 1

            if stage["name"] == "full_finetune" and epochs_without_improvement >= recipe["patience"]:
                print(f"      Early stopping triggered after {recipe['patience']} stale validation epochs.")
                break

        if stage["name"] == "full_finetune" and epochs_without_improvement >= recipe["patience"]:
            break

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    test_metrics = evaluate_topk_accuracy(
        model,
        test_loader,
        device=device,
        max_batches=args.max_eval_batches,
    )
    prediction_payload = collect_predictions(
        model,
        test_loader,
        device,
        max_batches=args.max_eval_batches,
    )
    confusion = build_confusion_matrix(
        prediction_payload["targets"],
        prediction_payload["predictions"],
        num_classes=num_classes,
    )
    per_class = per_class_accuracy_from_confusion(confusion, split_meta["class_names"])

    history_csv = run_dir / "history.csv"
    history_plot = run_dir / "training_curves.png"
    confusion_plot = run_dir / "confusion_matrix.png"
    metrics_file = run_dir / "metrics.json"
    per_class_file = run_dir / "per_class_accuracy.json"

    save_history_csv(history, history_csv)
    plot_training_history(history, history_plot)
    plot_confusion_matrix(confusion, split_meta["class_names"], confusion_plot)
    with open(per_class_file, "w", encoding="utf-8") as handle:
        json.dump(per_class, handle, indent=2)

    baseline_accuracy = load_baseline_accuracy(model_key)

    result_payload = {
        "model_key": model_key,
        "name": model_name,
        "strategy": recipe["strategy"],
        "seed": args.seed,
        "split_seed": split_meta["seed"],
        "input_size": input_size,
        "num_classes": num_classes,
        "recipe": recipe,
        "stage_plan": stage_plan,
        "model_info": model_info,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "best_stage": best_stage,
        "baseline_top1_accuracy": baseline_accuracy,
        "test_metrics": test_metrics,
        "per_class_accuracy": per_class,
        "artifacts": {
            "checkpoint": str(best_checkpoint.as_posix()),
            "history_csv": str(history_csv.as_posix()),
            "training_curves": str(history_plot.as_posix()),
            "confusion_matrix": str(confusion_plot.as_posix()),
            "per_class_accuracy": str(per_class_file.as_posix()),
            "recipe": str(recipe_file.as_posix()),
        },
        "elapsed_minutes": round((time.time() - started_at) / 60.0, 2),
    }

    if baseline_accuracy is not None:
        result_payload["accuracy_gain"] = round(test_metrics["top1_accuracy"] - baseline_accuracy, 2)

    with open(metrics_file, "w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2)

    print(f"    Best epoch         : {best_epoch} ({best_stage})")
    print(f"    Test accuracy      : {test_metrics['top1_accuracy']}% top-1")
    if baseline_accuracy is not None:
        print(f"    Accuracy lift      : +{result_payload['accuracy_gain']} pts vs. baseline")

    return result_payload


def write_summary(results: dict, split_meta: dict, device: str) -> None:
    leaderboard = sorted(
        results.values(),
        key=lambda payload: payload["test_metrics"]["top1_accuracy"],
        reverse=True,
    )
    accuracy_comparison = []
    for payload in leaderboard:
        if payload.get("baseline_top1_accuracy") is None:
            continue
        accuracy_comparison.append({
            "model_key": payload["model_key"],
            "name": payload["name"],
            "baseline_top1_accuracy": payload["baseline_top1_accuracy"],
            "finetuned_top1_accuracy": payload["test_metrics"]["top1_accuracy"],
            "gain": payload.get("accuracy_gain"),
        })

    summary_payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "device": device,
            "split_seed": split_meta["seed"],
            "split_counts": split_meta["counts"],
            "class_names": split_meta["class_names"],
        },
        "recommended_export_model": leaderboard[0]["model_key"] if leaderboard else None,
        "leaderboard": [{
            "model_key": payload["model_key"],
            "name": payload["name"],
            "top1_accuracy": payload["test_metrics"]["top1_accuracy"],
            "best_val_accuracy": payload["best_val_accuracy"],
            "gain": payload.get("accuracy_gain"),
        } for payload in leaderboard],
        "models": results,
        "accuracy_comparison": accuracy_comparison,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    model_config = load_model_config()
    recipe_book = load_recipe_book(args.recipe_file)

    if args.list_models:
        for model_key, model_cfg in model_config.items():
            print(f"{model_key:<22} {model_cfg['name']}")
        return

    missing_models = [model_key for model_key in args.models if model_key not in model_config]
    if missing_models:
        raise SystemExit(f"Unknown model keys: {', '.join(missing_models)}")

    device = resolve_device(args.device)
    seed_everything(args.seed)

    print()
    print("  ==========================================================")
    print("    EUROSAT FINE-TUNING PIPELINE")
    print("    Staged transfer learning for embedded vision models")
    print("  ==========================================================")
    print()
    print(f"  Device      : {device}")
    print(f"  Models      : {', '.join(args.models)}")
    print(f"  Data root   : {args.data_dir}")

    _, _, _, split_meta = get_eurosat_loaders(
        input_size=224,
        batch_size=8,
        num_workers=max(recipe_book.get('defaults', {}).get("num_workers", 4), 1),
        data_dir=args.data_dir,
        seed=args.seed,
    )
    counts = split_meta["counts"]
    print(f"  Split       : train={counts['train']}, val={counts['val']}, test={counts['test']}")
    print(f"  Classes     : {len(split_meta['class_names'])}")
    print()

    results = {}
    for model_key in args.models:
        recipe = resolve_recipe(model_key, args, recipe_book)
        results[model_key] = run_for_model(model_key, model_config, args, device, split_meta, recipe)

    write_summary(results, split_meta, device)

    print()
    print("  ------------------------------------------------")
    print(f"  Summary saved to {SUMMARY_FILE}")
    print("  Re-run python 02_generate_visualizations.py to refresh the dashboard.")
    print("  ------------------------------------------------")
    print()


if __name__ == "__main__":
    main()
