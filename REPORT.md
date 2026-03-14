# Efficient Vision Model Benchmark: Architecture Evaluation Report (2025)

## Overview
This report summarizes the findings from the **Efficient Vision Model Benchmark**, a project engineered to rigorously evaluate and compare SOTA computer vision architectures. The primary objective is to investigate the trade-offs between computational efficiency (latency, FLOPs, parameter count) and predictive performance (accuracy) under identical, carefully controlled evaluation conditions. 

In real-world applications (e.g., edge AI, drone-based vision, embedded systems), simply chasing raw accuracy is insufficient. A robust AI pipeline must account for hardware constraints, runtime budgets, and parameter footprints. 

## Methodology
To ensure reproducibility and fairness, all models were evaluated using the following standardized protocol:
* **Hardware**: Tested on CUDA-accelerated infrastructure (NVIDIA RTX 3060 Laptop GPU) and CPU fallbacks to assess edge-like constraints.
* **Dataset**: CIFAR-10 classification task (10,000 test images), with standardized ImageNet normalization and dynamic classifier head adaptation.
* **Architectures**: A curated selection spanning classical baselines (ResNet-50), mobile-first designs (MobileNetV3, EfficientNet), and modern SOTA approaches (ConvNeXt-Tiny, Swin-Tiny, MaxViT-Tiny, RegNetY).
* **Metrics Tracked**: 
    1. Top-1 Accuracy
    2. Inference Latency (ms) with proper warmup periods
    3. Multi-batch throughput
    4. Computational footprint (Parameters and MACs/FLOPs via `ptflops`)

## Key Findings

1. **The Modern edge ViTs (2024-2025)**: Models like `EfficientViT` and `FastViT-MCI4` effectively demonstrate how Apple and MIT researchers have squeezed Transformer-level accuracy into memory-bound edge environments. They are the new gold standard for UAVs or real-time dashcam deployments.
2. **Re-parameterized Speed (RepViT)**: The benchmarking of `RepViT` highlights a structural fusion of Transformers and CNNs that sustains massive throughput on non-optimized edge hardware (CPUs) without sacrificing representation power.
3. **Universal Architectures (MobileNetV4)**: The introduction of Google's 2024 `MobileNetV4` sets a new baseline for hardware abstraction, proving that classical mobile CNN topologies still push boundaries when intelligently scaled.

## Relevance to Applied ML & Data Pipelines
This repository serves as a blueprint for structured quantitative evaluation—a critical skill set for an **AI Software Engineer**. It demonstrates:
* **Modular Pipeline Design**: The codebase cleanly separates model registries (`yaml`), data loading (`torchvision`), measurement utilities, and visualization modules.
* **Robustness & Validation**: Emphasis on statistically sound latency measurements (warmups, medians, p95 bounds) rather than cherry-picked best runs.
* **Data-Driven Engineering**: Outputs are directly translated into actionable insights (e.g., Pareto-front visualizations) to allow stakeholders to make informed architecture decisions based on specific project constraints.

*This project is completely open-source. For detailed setup instructions and replication steps, please refer to the main repository `README.md`.*
