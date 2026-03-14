# EuroSAT Edge Vision Benchmark Report

## Executive Summary

This repository now has two distinct stages:

1. **Transfer baseline benchmark**  
   Pretrained backbones are adapted to EuroSAT by resetting the classifier head to 10 classes and evaluating immediately. This is useful for screening latency, throughput, and footprint, but the accuracy numbers should be interpreted as a lower-bound transfer baseline.

2. **Fine-tuning pipeline**  
   The new `03_finetune_models.py` script performs supervised transfer learning on deterministic EuroSAT train/val/test splits and stores checkpoints, curves, and summary metrics for the most relevant models.

That combination is much more defensible than the earlier framing, because it separates:

- raw deployment characteristics
- pre-adaptation baseline behavior
- post-adaptation performance

## Why The Earlier Framing Needed Correction

Replacing a pretrained classifier with a fresh 10-class head means the head starts random. Evaluating that head without training is not a true zero-shot transfer measurement. The previous README language overstated what the benchmark proved.

The corrected framing is:

- **Benchmark script:** head-reset transfer baseline
- **Training script:** the real EuroSAT adaptation stage

This matters because a reviewer with ML experience will notice the difference immediately.

## What Was Improved

### 1. Data handling

- deterministic EuroSAT train/val/test splits
- split metadata cached in `results/splits/`
- train-time aerial augmentations
- evaluation on a held-out test split instead of the full dataset

### 2. Visualization quality

The original plots had two practical problems:

- labels collided and became unreadable
- large outliers flattened the entire parameter comparison

The refreshed generator fixes this by using:

- a log-scale latency axis for the scatter plot
- a Pareto frontier overlay
- bubble sizes scaled by the square root of parameter count
- a log-scale parameter chart
- a CPU vs. GPU latency breakdown instead of the stale CIFAR-era radar chart

### 3. Recruiter-facing presentation

A new static site in `docs/` now acts as a GitHub Pages dashboard:

- concise benchmark narrative
- generated figures
- normalized model table
- explicit role alignment for the Quantum Systems position

This is a better artifact for a recruiter than a raw notebook or a README alone.

### 4. Fine-tuning capability

The new training pipeline supports:

- recommended default models:
  - `mobilevitv2_050`
  - `mobilenet_v3_large`
  - `convnext_tiny`
- `head` or `full` fine-tuning strategies
- early stopping on validation accuracy
- checkpoints, CSV logs, and training curves
- summary JSON for downstream dashboard updates

## Why These Models Are The Right Fine-Tune Targets

Based on the current baseline sweep:

- **MobileViTv2-0.5** offers the most attractive accuracy-per-parameter ratio.
- **MobileNetV3-Large** is the fastest CPU model in the practical cluster.
- **ConvNeXt-Tiny** gives a heavier pure-CNN reference point against the mobile hybrids.

This is a balanced trio for testing three useful hypotheses:

1. Does a tiny hybrid model adapt well enough to dominate the Pareto curve?
2. Does the latency winner stay compelling after supervised adaptation?
3. Does a larger CNN recover more strongly than the mobile models once trained on aerial data?

## Role Alignment With Quantum Systems

The live Quantum Systems job posting emphasizes:

- preparing/selecting data
- training and validating models
- deployment on embedded UxV platforms
- optimization for constrained hardware, including quantization/pruning/distillation
- hands-on edge deployment on Jetson, ARM, FPGA, or custom SoCs

That makes the most logical sequence:

1. benchmark deployment characteristics
2. fine-tune on aerial data
3. quantize and export the strongest model for embedded inference

## Recommended Next Step After Fine-Tuning

Once the fine-tuned results are generated, add:

**INT8 quantization + ONNX/TensorRT export**

That will produce the strongest end-to-end story for the target role:

- data pipeline
- model selection
- supervised adaptation
- deployment optimization

## Bottom Line

The repository is materially stronger now because it no longer relies on a misleading “zero-shot” story. It presents a cleaner and more realistic engineering sequence:

**screen models -> adapt them -> compare the lift -> prepare for edge deployment**
