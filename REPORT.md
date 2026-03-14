# EuroSAT Edge Vision Benchmark Report

## Executive Summary

This repository now demonstrates a complete, defensible embedded-vision workflow:

1. benchmark modern efficient backbones on a corrected EuroSAT split
2. fine-tune the strongest candidates with a staged transfer recipe
3. export the best deployment candidate to ONNX and INT8 ONNX
4. publish the outcome as a recruiter-facing web page and PDF packet

The result is substantially stronger than a standard "train a classifier" portfolio project because it shows the full engineering loop from dataset handling to deployment-oriented optimization.

## What Changed In This Pass

### 1. The EuroSAT cache was repaired

The local dataset copy used earlier was not the canonical EuroSAT RGB release. It contained only 9 classes and 23,008 samples. The pipeline now validates the cache explicitly and rejects incomplete local copies before benchmarking or training.

The corrected local dataset now matches the standard release:

- 27,000 RGB image patches
- 10 land-use classes
- deterministic split: 18,900 train / 4,050 val / 4,050 test

This matters because otherwise all benchmark and fine-tune numbers would be anchored to the wrong task definition.

### 2. The model registry was refreshed

The `RepViT-M0.9` timm identifier in the registry had drifted. It now resolves correctly against current timm weights, so the benchmark table no longer contains a missing row caused by a stale pretrained tag.

### 3. The fine-tuning recipe was upgraded

`03_finetune_models.py` is now a staged transfer-learning pipeline rather than a single fixed loop.

The default recipe uses:

- linear-probe warmup
- full fine-tuning with differential learning rates
- AdamW
- cosine decay with warmup
- AMP on CUDA
- label smoothing
- gradient clipping
- per-model recipe overrides via `configs/finetune_recipes.yaml`

It also writes:

- best checkpoint
- history CSV
- training curves
- confusion matrix
- per-class accuracy JSON
- summary JSON for downstream reporting

### 4. Deployment export was added

`04_export_deployment_artifacts.py` now exports the selected fine-tuned checkpoint as:

- FP32 ONNX
- validated ONNX outputs against PyTorch
- calibration dataset for PTQ
- INT8 QDQ ONNX
- optional TensorRT engines when `trtexec` is available

On this machine, TensorRT engine creation is implemented but skipped because local TensorRT tooling is not installed. That is reported explicitly in `results/deployment/summary.json`.

### 5. The project page was rebuilt

The `docs/` site now acts as a proper project page rather than a bare static dashboard:

- clear repo / report / PDF links
- fine-tune result section
- deployment artifact section
- print-specific link highlighting for recruiter PDFs
- GitHub Pages workflow for automatic publishing from `main`

## Measured Results

### Baseline benchmark

The corrected baseline run measures the head-reset transfer lower bound on the 10-class EuroSAT task.

Headline observations:

- **FastViT-MCI4** achieved the highest raw baseline top-1 accuracy at **13.41%**, but at an edge-inappropriate **385.84 ms CPU latency**
- **MobileNetV4-Small** is the fastest screened model at **15.62 ms CPU latency**
- **RepViT-M0.9** and **MobileNetV3-Large** are the most interesting practical baseline references once accuracy and latency are considered together

These baseline numbers are useful for **screening**, not for final model selection, because the classifier head is still untrained.

### Fine-tuned models

Held-out EuroSAT test accuracy after staged adaptation:

| Model | Test Top-1 | Gain vs. Baseline | Role in the portfolio |
|---|---:|---:|---|
| ConvNeXt-Tiny | 99.14% | +87.76 pts | absolute accuracy winner |
| MobileNetV3-Large | 98.74% | +88.67 pts | best deployment candidate |
| MobileViTv2-0.5 | 97.90% | +90.34 pts | strongest tiny-model transfer result |

These results validate the central engineering claim of the project:

**top-down aerial imagery needs supervised adaptation, but modern pretrained backbones can recover to near-ceiling performance once the domain shift is addressed properly**

## Why MobileNetV3-Large Was Chosen For Export

ConvNeXt-Tiny achieved the highest accuracy, but it is not the most practical embedded deployment target in this set.

`MobileNetV3-Large` was selected for deployment export because it is:

- within **0.40 percentage points** of the best fine-tuned accuracy
- materially faster in the baseline latency profile
- much smaller and more realistic for edge packaging

This is the more mature engineering decision for a UAV-facing portfolio artifact. Accuracy alone is not the correct optimization target.

## Deployment Export Outcome

The deployment stage produced:

- FP32 ONNX: `results/deployment/mobilenet_v3_large/mobilenet_v3_large_fp32.onnx`
- INT8 QDQ ONNX: `results/deployment/mobilenet_v3_large/mobilenet_v3_large_int8_qdq.onnx`
- calibration data: `results/deployment/mobilenet_v3_large/calibration_data.npz`

Measured artifact sizes:

- FP32 ONNX: **16.07 MB**
- INT8 QDQ ONNX: **4.25 MB**

That is roughly a **3.8x footprint reduction** from the ONNX export path alone.

The script also validates ONNX outputs against PyTorch. For the exported MobileNetV3-Large checkpoint, the recorded difference was small:

- max absolute diff: **0.003182**
- mean absolute diff: **0.001391**

## Presentation Layer

The static site and PDF now do real work:

- explain the corrected methodology
- show the baseline and fine-tuned visuals
- surface the deployment candidate explicitly
- provide direct recruiter navigation to the repo, report, role posting, and PDF

The PDF export also uses print-specific hyperlink styling so the links remain visible and scannable after export.

## Remaining Limitation

The one substantive environment limitation left is **local TensorRT availability**.

The export path is already implemented to build TensorRT engines automatically, but on this machine:

- `trtexec` is not installed
- TensorRT engine files were therefore not generated

That should be presented honestly. The repo now shows the correct behavior: export ONNX and INT8 ONNX successfully, and report TensorRT as skipped rather than fabricating an engine artifact.

## Bottom Line

The repository is now in a much stronger state than the original benchmark:

- the data pipeline is validated
- the baseline is rerun on the correct task
- the fine-tuning results are real and strong
- the export stage produces usable deployment artifacts
- the public-facing site and PDF are recruiter-ready

As a job application artifact, the project now reads like an internal edge-vision evaluation tool, which is exactly the right direction for the target Quantum Systems role.
