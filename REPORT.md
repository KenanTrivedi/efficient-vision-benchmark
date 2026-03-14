# EuroSAT SWaP Trade-off Evaluation: Modern Edge Vision Architectures

## Executive Summary

The objective of this ongoing benchmarking suite is to empirically evaluate modern state-of-the-art (SOTA) computer vision architectures—specifically focusing on lightweight Edge Convolutional and Vision Transformer (ViT) hybrids released between 2023 and 2025. 

Unlike traditional generic benchmarks (e.g., CIFAR/ImageNet), this evaluation explicitly targets the **Size, Weight, and Power (SWaP) constraints** inherent to Unmanned Aerial Vehicles (UAVs) and autonomous robotics via the **EuroSAT Aerial dataset**. By controlling all preprocessing variables and mapping evaluations to domain-specific drone sensor data, we provide a deeply rigorous Pareto frontier for inference latency, memory scaling, and representation capability.

This infrastructure is critical for mapping production AI to embedded hardware units (like NVIDIA Jetson Orin Nano, ARM SoCs, or Hexagon DSPs). 

---

## Benchmarking Methodology & Rigor

Establishing a scientifically valid abstraction layer requires strictly controlled dependent and independent variables. Relying on paper-published numbers is flawed due to variations in augmentations, data regimes (e.g., ImageNet-12k vs 1k), hardware architectures, and compiler stacks. 

### Variables and Control Mechanisms
* **Domain Target (Dataset):** The EuroSAT satellite/aerial dataset. Simulates real-world top-down mapping anomalies and terrain spatial topologies inherently captured by UAV sensors.
* **Transform Consistency:** Identical spatial scaling (usually 224x224), center-cropping, and `ImageNet` channel standard deviation normalizations applied to all architectures prior to feature extraction.
* **Telemetry Gathering:** 
  * **Latency (ms):** Benchmarked via rigorous warmup loops (rejecting OS jitter) and median sampling over $N=50$ strict isolated single-batch inference passes. 
  * **Footprint:** Extracting static parameter counts mapping to physical disk buffer constraints. 
  * **Throughput (ms/img):** Calculating inference degradation dynamics as batch sizes scale from $1 \to 16$.
  * **MACs/FLOPs:** Extracted directly using structural analysis (`ptflops`), representing the strictly upper-bound energy expenditure profile independent of hardware capabilities.

---

## 2025 Empirical Findings & Hardware Implications

By evaluating modern topologies (like `MobileNetV4` and `FastViT-MCI4`) against historical standards (`ResNet-50`), several critical architectural choices emerge for Edge ML Data Pipelines:

### 1. The Multi-Axis Attention Efficiency Ceiling
While foundational global-attention transformers revolutionized representation, they scale quadratically with memory. The introduction of models like **FastViT-MCI4 (Apple, 2025)** and **EfficientViT (MIT, 2023)** validate that *Linear/Sub-quadratic Attention* pathways successfully map highly complex aerial feature topologies (like dense foliage vs. residential density) without shattering the memory caches of edge architectures. They represent the current Pareto-optimal point for modern UAV analytics.

### 2. Re-parameterized CNN Survivability
Despite the shift toward ViTs, pure convolutional logic remains critically fast when compiled via TensorRT. The **RepViT** (Tsinghua, 2024) and **MobileNetV4** (Google, 2024) models demonstrate a structural fusion approach. By utilizing multi-scale convolutions designed specifically for hardware abstraction, they provide phenomenal sub-5ms CPU latency while rivaling standard ViT representation boundaries over the EuroSAT domain.

### 3. Context for Applied AI Engineering
The results from this repository establish the foundation required for the final deployment pipelines: **Quantization and Network Compilation.**
Once a Pareto-optimal model is selected based on this benchmarking suite (minimizing FLOPs while maximizing Top-1 Accuracy), the logical next step is driving it through Post-Training Quantization (PTQ) to `INT8` and exporting to `.onnx` for native C++ device execution.

---
*Authored by Kenan Radheshyam Trivedi.*
*Developed as an independent demonstration of Edge ML deployment principles and autonomous system architecture analysis.*
