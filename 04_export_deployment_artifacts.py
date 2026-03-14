#!/usr/bin/env python3
"""
04_export_deployment_artifacts.py - Deployment export pipeline
==============================================================
Exports the best fine-tuned EuroSAT model to ONNX, prepares INT8 calibration
data, optionally quantizes the ONNX graph, and attempts TensorRT engine builds
when local tooling is available.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from src.data import DEFAULT_SEED, get_eurosat_loaders
from src.models import get_model, load_model_config
from src.training import resolve_device


FINETUNE_SUMMARY_FILE = Path("results/finetune/summary.json")
DEPLOYMENT_RESULTS_DIR = Path("results/deployment")
DEPLOYMENT_SUMMARY_FILE = DEPLOYMENT_RESULTS_DIR / "summary.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fine-tuned EuroSAT models for deployment.")
    parser.add_argument("--model-key", default=None, help="Model key to export. Defaults to the best fine-tuned model.")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path. Overrides summary lookup.")
    parser.add_argument("--summary-file", default=str(FINETUNE_SUMMARY_FILE), help="Fine-tune summary JSON.")
    parser.add_argument("--output-dir", default=str(DEPLOYMENT_RESULTS_DIR), help="Deployment artifact directory.")
    parser.add_argument("--data-dir", default="./data", help="EuroSAT root directory.")
    parser.add_argument("--device", default=None, help="Export device. Defaults to CUDA when available.")
    parser.add_argument("--batch-size", type=int, default=1, help="Export batch size.")
    parser.add_argument("--calibration-samples", type=int, default=256, help="Number of validation samples used for INT8 calibration.")
    parser.add_argument("--calibration-batch-size", type=int, default=32, help="Batch size used when collecting calibration data.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed / split seed.")
    parser.add_argument("--opset", type=int, default=19, help="ONNX opset version.")
    parser.add_argument("--skip-int8", action="store_true", help="Skip INT8 ONNX quantization.")
    parser.add_argument("--skip-trt", action="store_true", help="Skip TensorRT engine export.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_export_target(args: argparse.Namespace) -> dict:
    summary_payload = load_json(Path(args.summary_file))
    model_key = args.model_key or summary_payload.get("recommended_export_model")
    if not model_key:
        leaderboard = summary_payload.get("leaderboard", [])
        if not leaderboard:
            raise SystemExit("No fine-tuned models were found in the summary file.")
        model_key = leaderboard[0]["model_key"]

    model_payload = summary_payload.get("models", {}).get(model_key)
    if not model_payload:
        raise SystemExit(f"Model '{model_key}' was not found in the fine-tune summary.")

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(model_payload["artifacts"]["checkpoint"])
    return {
        "summary": summary_payload,
        "model_key": model_key,
        "model_payload": model_payload,
        "checkpoint_path": checkpoint_path,
    }


def load_model_for_export(checkpoint_path: Path, device: str) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_key = checkpoint["model_key"]
    class_names = checkpoint.get("class_names", [])
    num_classes = len(class_names)
    input_size = checkpoint.get("input_size") or load_model_config()[model_key].get("input_size", 224)

    model = get_model(model_key, num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    return model, {
        "model_key": model_key,
        "model_name": checkpoint["model_name"],
        "class_names": class_names,
        "input_size": input_size,
        "recipe": checkpoint.get("recipe", {}),
        "best_val_accuracy": checkpoint.get("val_accuracy"),
        "checkpoint_epoch": checkpoint.get("epoch"),
    }


def export_onnx_model(
    model: torch.nn.Module,
    *,
    input_size: int,
    batch_size: int,
    device: str,
    output_path: Path,
    opset: int,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.randn(batch_size, 3, input_size, input_size, device=device)
    metadata = {
        "path": str(output_path.as_posix()),
        "opset": opset,
        "dynamo": False,
        "dynamic_batch": True,
    }

    try:
        batch_dim = torch.export.Dim("batch")
        torch.onnx.export(
            model,
            (sample,),
            str(output_path),
            input_names=["images"],
            output_names=["logits"],
            opset_version=opset,
            dynamo=True,
            dynamic_shapes=({0: batch_dim},),
        )
        metadata["dynamo"] = True
    except Exception:
        torch.onnx.export(
            model,
            sample,
            str(output_path),
            input_names=["images"],
            output_names=["logits"],
            opset_version=opset,
            dynamo=False,
            dynamic_axes={
                "images": {0: "batch"},
                "logits": {0: "batch"},
            },
        )

    return metadata


def validate_onnx_export(model: torch.nn.Module, onnx_path: Path, input_size: int, device: str) -> dict | None:
    if importlib.util.find_spec("onnxruntime") is None:
        return None

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    sample = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
    with torch.no_grad():
        torch_output = model(torch.from_numpy(sample).to(device)).detach().cpu().numpy()
    onnx_output = session.run(None, {"images": sample})[0]

    return {
        "max_abs_diff": round(float(np.max(np.abs(torch_output - onnx_output))), 6),
        "mean_abs_diff": round(float(np.mean(np.abs(torch_output - onnx_output))), 6),
    }


def save_calibration_data(
    *,
    input_size: int,
    data_dir: str,
    output_path: Path,
    num_samples: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> dict:
    _, val_loader, _, split_meta = get_eurosat_loaders(
        input_size=input_size,
        batch_size=batch_size,
        num_workers=num_workers,
        data_dir=data_dir,
        seed=seed,
    )

    collected_images = []
    collected_labels = []
    sample_count = 0
    for images, labels in val_loader:
        collected_images.append(images.numpy())
        collected_labels.append(labels.numpy())
        sample_count += len(labels)
        if sample_count >= num_samples:
            break

    image_array = np.concatenate(collected_images, axis=0)[:num_samples].astype(np.float32)
    label_array = np.concatenate(collected_labels, axis=0)[:num_samples].astype(np.int64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, images=image_array, labels=label_array)

    return {
        "path": str(output_path.as_posix()),
        "num_samples": int(image_array.shape[0]),
        "class_names": split_meta["class_names"],
    }


def quantize_int8_onnx(
    fp32_onnx_path: Path,
    calibration_path: Path,
    output_path: Path,
) -> dict:
    if importlib.util.find_spec("onnxruntime.quantization") is None:
        return {
            "status": "skipped",
            "reason": "onnxruntime quantization is not installed.",
        }

    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    import onnxruntime as ort

    calibration_data = np.load(calibration_path)["images"]
    input_name = ort.InferenceSession(str(fp32_onnx_path), providers=["CPUExecutionProvider"]).get_inputs()[0].name

    class NpzCalibrationReader(CalibrationDataReader):
        def __init__(self, input_name: str, images: np.ndarray):
            self.input_name = input_name
            self.images = images
            self.index = 0

        def get_next(self):
            if self.index >= len(self.images):
                return None
            payload = {self.input_name: self.images[self.index:self.index + 1]}
            self.index += 1
            return payload

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=str(fp32_onnx_path),
        model_output=str(output_path),
        calibration_data_reader=NpzCalibrationReader(input_name, calibration_data),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )

    return {
        "status": "ok",
        "path": str(output_path.as_posix()),
        "backend": "onnxruntime-static-qdq",
    }


def try_build_tensorrt_engine(onnx_path: Path, output_path: Path, extra_flags: list[str] | None = None) -> dict:
    trtexec_path = shutil.which("trtexec")
    if not trtexec_path:
        return {
            "status": "skipped",
            "reason": "trtexec was not found on PATH.",
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={output_path}",
        "--skipInference",
    ]
    if extra_flags:
        command.extend(extra_flags)

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    log_path = output_path.with_suffix(".log.txt")
    log_path.write_text(result.stdout + "\n\n" + result.stderr, encoding="utf-8")

    if result.returncode != 0:
        return {
            "status": "failed",
            "reason": f"trtexec exited with code {result.returncode}",
            "log": str(log_path.as_posix()),
        }

    return {
        "status": "ok",
        "path": str(output_path.as_posix()),
        "log": str(log_path.as_posix()),
    }


def add_file_size_metadata(payload: dict) -> dict:
    path = payload.get("path")
    if path and Path(path).exists():
        payload["size_mb"] = round(Path(path).stat().st_size / (1024 ** 2), 2)
    return payload


def main() -> None:
    args = parse_args()
    started_at = time.time()
    device = resolve_device(args.device)
    target = resolve_export_target(args)
    output_dir = Path(args.output_dir) / target["model_key"]
    output_dir.mkdir(parents=True, exist_ok=True)

    model, model_meta = load_model_for_export(target["checkpoint_path"], device=device)
    model_config = load_model_config()[target["model_key"]]

    print()
    print("  ==========================================================")
    print("    DEPLOYMENT EXPORT PIPELINE")
    print("    ONNX + INT8 quantization + TensorRT-ready artifacts")
    print("  ==========================================================")
    print()
    print(f"  Model       : {model_meta['model_name']} ({target['model_key']})")
    print(f"  Device      : {device}")
    print(f"  Checkpoint  : {target['checkpoint_path']}")
    print()

    fp32_onnx_path = output_dir / f"{target['model_key']}_fp32.onnx"
    int8_onnx_path = output_dir / f"{target['model_key']}_int8_qdq.onnx"
    calibration_path = output_dir / "calibration_data.npz"
    fp16_engine_path = output_dir / f"{target['model_key']}_fp16.engine"
    int8_engine_path = output_dir / f"{target['model_key']}_int8.engine"

    print("  [1] Exporting ONNX graph ...", end=" ", flush=True)
    onnx_export = export_onnx_model(
        model,
        input_size=model_meta["input_size"],
        batch_size=args.batch_size,
        device=device,
        output_path=fp32_onnx_path,
        opset=args.opset,
    )
    onnx_export = add_file_size_metadata(onnx_export)
    print("OK")

    print("  [2] Validating ONNX numerics ...", end=" ", flush=True)
    onnx_validation = validate_onnx_export(model, fp32_onnx_path, model_meta["input_size"], device=device)
    print("OK" if onnx_validation else "SKIPPED")

    print("  [3] Saving calibration data ...", end=" ", flush=True)
    calibration_payload = save_calibration_data(
        input_size=model_meta["input_size"],
        data_dir=args.data_dir,
        output_path=calibration_path,
        num_samples=args.calibration_samples,
        batch_size=args.calibration_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    calibration_payload = add_file_size_metadata(calibration_payload)
    print("OK")

    if args.skip_int8:
        int8_payload = {
            "status": "skipped",
            "reason": "INT8 export disabled by --skip-int8.",
        }
    else:
        print("  [4] Quantizing INT8 ONNX ...", end=" ", flush=True)
        int8_payload = quantize_int8_onnx(fp32_onnx_path, calibration_path, int8_onnx_path)
        int8_payload = add_file_size_metadata(int8_payload)
        print("OK" if int8_payload.get("status") == "ok" else "SKIPPED")

    if args.skip_trt:
        fp16_engine = {"status": "skipped", "reason": "TensorRT export disabled by --skip-trt."}
        int8_engine = {"status": "skipped", "reason": "TensorRT export disabled by --skip-trt."}
    else:
        print("  [5] Building FP16 TensorRT engine ...", end=" ", flush=True)
        fp16_engine = try_build_tensorrt_engine(fp32_onnx_path, fp16_engine_path, extra_flags=["--fp16"])
        fp16_engine = add_file_size_metadata(fp16_engine)
        print("OK" if fp16_engine.get("status") == "ok" else "SKIPPED")

        if int8_payload.get("status") == "ok":
            print("  [6] Building INT8 TensorRT engine ...", end=" ", flush=True)
            int8_engine = try_build_tensorrt_engine(int8_onnx_path, int8_engine_path)
            int8_engine = add_file_size_metadata(int8_engine)
            print("OK" if int8_engine.get("status") == "ok" else "SKIPPED")
        else:
            int8_engine = {
                "status": "skipped",
                "reason": "INT8 ONNX export was not available.",
            }

    summary_payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "device": device,
            "elapsed_minutes": round((time.time() - started_at) / 60.0, 2),
            "onnxruntime_available": importlib.util.find_spec("onnxruntime") is not None,
            "trtexec_available": shutil.which("trtexec") is not None,
        },
        "selected_model": {
            "model_key": target["model_key"],
            "name": model_meta["model_name"],
            "checkpoint": str(target["checkpoint_path"].as_posix()),
            "input_size": model_meta["input_size"],
            "test_top1_accuracy": target["model_payload"]["test_metrics"]["top1_accuracy"],
            "best_val_accuracy": model_meta["best_val_accuracy"],
            "checkpoint_epoch": model_meta["checkpoint_epoch"],
            "family": model_config.get("description"),
        },
        "artifacts": {
            "onnx_fp32": onnx_export,
            "onnx_validation": onnx_validation,
            "calibration_data": calibration_payload,
            "onnx_int8": int8_payload,
            "tensorrt_fp16": fp16_engine,
            "tensorrt_int8": int8_engine,
        },
    }

    DEPLOYMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DEPLOYMENT_SUMMARY_FILE, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    print()
    print("  ------------------------------------------------")
    print(f"  Deployment summary saved to {DEPLOYMENT_SUMMARY_FILE}")
    print("  ------------------------------------------------")
    print()


if __name__ == "__main__":
    main()
