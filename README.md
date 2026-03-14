# Aerial Vision Model Benchmark

> How fast and accurate are modern vision backbones when the camera looks *down*?

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-HuggingFace-yellow.svg)](https://huggingface.co/docs/timm)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Why This Project Exists

Most vision model benchmarks report numbers on ImageNet or CIFAR — datasets of everyday objects photographed at eye level. But a growing number of real-world applications operate from an **aerial perspective**: satellite monitoring, agricultural mapping, search-and-rescue drones, infrastructure inspection, and autonomous UAV navigation.

The visual statistics of top-down imagery are fundamentally different: textures dominate over shapes, rotational invariance matters, and objects lack the usual "gravity bias" of ground-level photos. A model that excels on ImageNet may behave very differently on overhead scenes.

This project asks a simple question:

> **Which lightweight, edge-deployable architectures generalise best to aerial imagery — and at what cost in latency and memory?**

To answer it, we benchmark **9 architectures** (from 2016-era ResNets to 2025-era hybrid ViTs) on the **EuroSAT** dataset under strictly controlled, reproducible conditions.

---

## The Dataset: EuroSAT

[EuroSAT](https://github.com/phelber/EuroSAT) ([Helber et al., 2019](https://ieeexplore.ieee.org/document/8736785)) is a land-use classification dataset derived from ESA Sentinel-2 satellite imagery. It contains **27,000 geo-referenced images** across **10 classes** (e.g., Highway, Industrial, Forest, Residential, River, AnnualCrop).

* Resolution: 64×64 RGB patches (upscaled to 224×224 for model compatibility)
* Distribution: real-world geographic imagery from across Europe
* Auto-downloaded via `torchvision.datasets.EuroSAT`

**Why EuroSAT?** It provides a standardised, publicly available proxy for the kind of top-down classification tasks encountered in aerial robotics, without requiring proprietary drone footage or restrictive licenses.

---

## Architectures Under Test

We selected a diverse set spanning four generations of efficient vision design:

| Architecture | Year | Source | Key Innovation | Reference |
|---|---|---|---|---|
| **ResNet-50** | 2016 | torchvision | Skip connections — the original deep baseline | [He et al., CVPR 2016](https://arxiv.org/abs/1512.03385) |
| **MobileNetV3-L** | 2019 | torchvision | NAS + squeeze-excite for mobile | [Howard et al., ICCV 2019](https://arxiv.org/abs/1905.02244) |
| **EfficientNet-B0** | 2019 | torchvision | Compound scaling | [Tan & Le, ICML 2019](https://arxiv.org/abs/1905.11946) |
| **ConvNeXt-Tiny** | 2022 | torchvision | Modernised pure-CNN, ViT-competitive | [Liu et al., CVPR 2022](https://arxiv.org/abs/2201.03545) |
| **MobileViTv2-0.5** | 2022 | timm | Separable self-attention | [Mehta et al., T-PAMI 2023](https://arxiv.org/abs/2206.02680) |
| **EfficientViT-M0** | 2023 | timm | Linear attention for memory-bound hardware | [Cai et al., ICCV 2023](https://arxiv.org/abs/2305.07027) |
| **RepViT-M0.9** | 2024 | timm | Re-parameterised CNN-ViT hybrid | [Wang et al., CVPR 2024](https://arxiv.org/abs/2307.09283) |
| **MobileNetV4-S** | 2024 | timm | Universal hardware abstraction | [Qin et al., CVPR 2024](https://arxiv.org/abs/2404.10518) |
| **FastViT-MCI4** | 2025 | timm | Latest Apple hybrid ViT iteration | [Apple ML Research, 2025](https://github.com/apple/ml-fastvit) |

---

## What We Measure

For each model, under identical conditions:

| Metric | Method | Why It Matters |
|---|---|---|
| **Top-1 / Top-5 Accuracy** | Forward pass on full EuroSAT set | How well does the model generalise to aerial textures? |
| **CPU Latency** | `time.perf_counter`, 20 runs, 5 warmup | Simulates inference on embedded ARM / Jetson without GPU |
| **GPU Latency** | `torch.cuda.synchronize`, 30 runs, 5 warmup | Best-case latency on accelerated hardware |
| **Throughput** | Images/second at batch sizes 1 and 16 | Real-time pipeline capacity |
| **MACs / FLOPs** | `ptflops` structural analysis | Upper bound on energy consumption per inference |
| **Parameter Count** | Sum of `model.parameters()` | Determines flash/RAM footprint on the device |

---

## Quickstart — Reproduce in 3 Steps

### Prerequisites

* Python ≥ 3.9
* NVIDIA GPU with CUDA (recommended, but CPU works too)
* ~500 MB disk space for model weights + dataset

### Step 1 — Clone & Install

```bash
git clone https://github.com/KenanTrivedi/efficient-vision-benchmark.git
cd efficient-vision-benchmark

# Create a virtual environment (or use conda)
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### Step 2 — Run the Benchmark

```bash
python 01_run_benchmark.py
```

The script will:
1. Ask you to select a compute device (CUDA or CPU)
2. Show you all available architectures and let you pick one or all
3. Auto-download the EuroSAT dataset on first run (~90 MB)
4. Evaluate each model and print structured results to the terminal
5. Save all telemetry to `results/benchmark_results.json`

### Step 3 — Generate Visualisations

```bash
python 02_generate_visualizations.py
```

Generates publication-ready charts in `results/figures/`:
* **Accuracy vs. Latency scatter** — shows the Pareto frontier
* **Model Size comparison** — horizontal bar chart of parameter counts

---

## Project Structure

```
efficient-vision-benchmark/
├── 01_run_benchmark.py           # Interactive benchmark runner
├── 02_generate_visualizations.py # Chart generator
├── configs/
│   └── models.yaml               # Architecture registry with paper refs
├── src/
│   ├── models.py                 # Model loading (timm + torchvision)
│   ├── data.py                   # EuroSAT data pipeline
│   ├── timing.py                 # Latency & throughput measurement
│   └── metrics.py                # Accuracy evaluation, FLOPs estimation
├── results/
│   ├── benchmark_results.json    # Raw telemetry (auto-generated)
│   └── figures/                  # Charts (auto-generated)
├── REPORT.md                     # Extended analysis & discussion
├── requirements.txt
└── LICENSE
```

---

## Future Work & Roadmap

- [ ] **ONNX / TensorRT export** — compile the best model into an optimised inference engine for embedded deployment
- [ ] **INT8 post-training quantisation** — measure accuracy degradation after 4× size reduction
- [ ] **Fine-tuning comparison** — linear probe vs. full fine-tune on EuroSAT to measure transfer efficiency
- [ ] **VisDrone dataset** — extend evaluation to real drone footage with object detection targets
- [ ] **Latency profiling on NVIDIA Jetson Orin Nano** — measure actual edge hardware performance
- [ ] **GitHub Pages dashboard** — interactive web viewer for benchmark results

---

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
2. Howard, A., et al. (2019). *Searching for MobileNetV3.* ICCV 2019. [arXiv:1905.02244](https://arxiv.org/abs/1905.02244)
3. Tan, M. & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling.* ICML 2019. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
4. Liu, Z., et al. (2022). *A ConvNet for the 2020s.* CVPR 2022. [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
5. Mehta, S. & Rastegari, M. (2023). *Separable Self-attention for Mobile Vision Transformers.* T-PAMI. [arXiv:2206.02680](https://arxiv.org/abs/2206.02680)
6. Cai, H., et al. (2023). *EfficientViT: Lightweight Multi-Scale Attention for High-Resolution Dense Prediction.* ICCV 2023. [arXiv:2305.07027](https://arxiv.org/abs/2305.07027)
7. Wang, A., et al. (2024). *RepViT: Revisiting Mobile CNN From ViT Perspective.* CVPR 2024. [arXiv:2307.09283](https://arxiv.org/abs/2307.09283)
8. Qin, D., et al. (2024). *MobileNetV4 — Universal Models for the Mobile Ecosystem.* CVPR 2024. [arXiv:2404.10518](https://arxiv.org/abs/2404.10518)
9. Helber, P., et al. (2019). *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS. [DOI:10.1109/JSTARS.2019.2918242](https://ieeexplore.ieee.org/document/8736785)

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built by [Kenan Radheshyam Trivedi](https://kenantrivedi.com) as an independent exploration of edge-efficient computer vision for aerial applications.*
