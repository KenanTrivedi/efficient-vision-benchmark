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

1. **The Modern CNN Revival**: `ConvNeXt-Tiny` successfully bridges the gap between traditional CNN efficiency and Vision Transformer (ViT) performance scale. For tasks where pure CNN topologies are preferred (due to hardware compiler support or legacy deployment pipelines), it offers an excellent accuracy/latency trade-off without the self-attention overhead.
2. **Transformer Trade-offs**: While models like `Swin-Tiny` and `MaxViT-Tiny` achieve dominant accuracy figures, their latency footprint on non-optimized hardware can be significantly larger than comparably-sized CNNs. They remain robust choices for high-end edge devices (e.g., NVIDIA Jetson Orin) where tensor cores can accelerate attention mechanisms.
3. **Sustained Mobile Excellence**: The `MobileNetV3` and `EfficientNet` families continue to provide the best pure "compute-to-accuracy" ratio for heavily constrained environments. `EfficientNetV2-S`, in particular, demonstrates the power of Fused-MBConv layers for minimizing memory bottlenecks during inference.

## Relevance to Applied ML & Data Pipelines
This repository serves as a blueprint for structured quantitative evaluation—a critical skill set for an **AI Software Engineer**. It demonstrates:
* **Modular Pipeline Design**: The codebase cleanly separates model registries (`yaml`), data loading (`torchvision`), measurement utilities, and visualization modules.
* **Robustness & Validation**: Emphasis on statistically sound latency measurements (warmups, medians, p95 bounds) rather than cherry-picked best runs.
* **Data-Driven Engineering**: Outputs are directly translated into actionable insights (e.g., Pareto-front visualizations) to allow stakeholders to make informed architecture decisions based on specific project constraints.

*This project is completely open-source. For detailed setup instructions and replication steps, please refer to the main repository `README.md`.*
