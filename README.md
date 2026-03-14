# EuroSAT Edge Vision Benchmark

> Benchmarking lightweight vision backbones for top-down imagery, then adapting the strongest candidates with reproducible transfer learning.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-HuggingFace-yellow.svg)](https://huggingface.co/docs/timm)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What This Repository Demonstrates

This repository is designed as an **edge-AI engineering artifact**, not a toy image-classification demo. It evaluates modern efficient backbones on **EuroSAT**, a public aerial/satellite benchmark, and then provides a **transfer-learning pipeline** for adapting the best models to the target domain under realistic deployment constraints:

- **Latency** on CPU and GPU
- **Throughput** at batch sizes 1 and 16
- **Parameter footprint** and **MACs**
- **Held-out EuroSAT accuracy**
- **Reproducible fine-tuning** with checkpoints, curves, and summary outputs

This is the workflow that matters for embedded UAV vision: measure first, then adapt, then prepare for deployment.

---

## Important Methodology Note

The current `01_run_benchmark.py` benchmark is **not a true zero-shot experiment**.

It evaluates a **head-reset transfer baseline**:

1. Load ImageNet-pretrained weights
2. Replace the classifier with a fresh 10-class EuroSAT head
3. Measure latency/footprint immediately
4. Evaluate the untrained head on the EuroSAT test split

That means the baseline accuracy is a **lower bound**, not a claim about genuine zero-shot transfer. The low numbers are expected because the new head has not been trained yet. The repository now makes this explicit and includes `03_finetune_models.py` to perform the actual adaptation step.

---

## Why EuroSAT

[EuroSAT](https://github.com/phelber/EuroSAT) ([Helber et al., 2019](https://ieeexplore.ieee.org/document/8736785)) contains **27,000 Sentinel-2 image patches** across **10 land-use classes** such as `Highway`, `Industrial`, `Residential`, and `River`.

It is a good public proxy for the visual conditions that matter in aerial robotics:

- top-down geometry instead of front-facing perspective
- texture-heavy scene understanding
- rotation tolerance
- domain shift relative to consumer-photo datasets

---

## Models Under Test

The registry spans classical baselines and modern mobile architectures:

| Model | Family | Year | Source |
|---|---|---:|---|
| ResNet-50 | residual CNN baseline | 2016 | torchvision |
| MobileNetV3-Large | mobile CNN | 2019 | torchvision |
| EfficientNet-B0 | compound-scaled CNN | 2019 | torchvision |
| ConvNeXt-Tiny | modernized CNN | 2022 | torchvision |
| MobileViTv2-0.5 | mobile ViT hybrid | 2022 | timm |
| EfficientViT-M0 | efficient attention | 2023 | timm |
| RepViT-M0.9 | re-parameterized mobile hybrid | 2024 | timm |
| MobileNetV4-Small | latest mobile CNN | 2024 | timm |
| FastViT-MCI4 | large hybrid baseline | 2025 | timm |

---

## Repository Workflow

### 1. Run the transfer baseline benchmark

```bash
python 01_run_benchmark.py
```

Outputs:

- `results/benchmark_results.json`
- latency, throughput, accuracy, footprint telemetry per model

### 2. Generate plots and GitHub Pages assets

```bash
python 02_generate_visualizations.py
```

Outputs:

- `results/figures/accuracy_vs_latency.png`
- `results/figures/parameter_footprint.png`
- `results/figures/cpu_gpu_latency_breakdown.png`
- `docs/index.html`
- `docs/site_data.json`
- `docs/site_data.js`

### 3. Fine-tune the recommended models

```bash
python 03_finetune_models.py
```

Defaults to the most interesting adaptation targets from the baseline:

- `mobilevitv2_050`
- `mobilenet_v3_large`
- `convnext_tiny`

Outputs:

- `results/finetune/<model_key>/best.pt`
- `results/finetune/<model_key>/history.csv`
- `results/finetune/<model_key>/training_curves.png`
- `results/finetune/<model_key>/metrics.json`
- `results/finetune/summary.json`

After fine-tuning completes, re-run:

```bash
python 02_generate_visualizations.py
```

to refresh the dashboard with the fine-tune comparison figure.

---

## Current Baseline Snapshot

The existing checked-in results come from the **head-reset transfer baseline** on the held-out EuroSAT test split.

| Model | Top-1 (%) | CPU Latency (ms) | Params |
|---|---:|---:|---:|
| ResNet-50 | 12.98 | 85.38 | 23.5M |
| ConvNeXt-Tiny | 12.28 | 58.32 | 27.8M |
| MobileViTv2-0.5 | 11.73 | 46.72 | 1.1M |
| MobileNetV3-Large | 9.94 | 31.32 | 4.2M |
| EfficientViT-M0 | 8.86 | 73.27 | 2.2M |
| MobileNetV4-Small | 6.89 | 32.46 | 2.5M |
| EfficientNet-B0 | 5.61 | 40.26 | 4.0M |
| FastViT-MCI4 | 3.36 | 898.06 | 318.7M |

These numbers are useful for **latency and footprint screening**, but the scientific story only becomes complete after running the fine-tuning stage.

### Accuracy vs. CPU Latency

![Accuracy vs Latency](results/figures/accuracy_vs_latency.png)

### Parameter Footprint

![Parameter Footprint](results/figures/parameter_footprint.png)

### CPU vs. GPU Latency

![CPU vs GPU Latency](results/figures/cpu_gpu_latency_breakdown.png)

---

## GitHub Pages Dashboard

The repo now includes a recruiter-friendly static site in [`docs/`](docs/) that summarizes:

- the benchmark protocol
- the latest generated figures
- a normalized comparison table
- the recommended fine-tune targets
- explicit alignment to the Quantum Systems AI Software Engineer role

Once GitHub Pages is enabled for the `docs/` folder, the dashboard can be used as a lightweight portfolio page and exported to PDF.

---

## Recommended Next Step

The strongest next technical addition after fine-tuning is:

**INT8 quantization + ONNX/TensorRT export of the best adapted model**

Why:

- the live Quantum Systems posting explicitly calls out deployment on embedded UxV hardware
- it also highlights optimization for resource-constrained environments, including **quantization, pruning, and distillation**
- once EuroSAT adaptation is demonstrated, deployment optimization is the most credible follow-up

Official role: [Quantum Systems - AI Software Engineer (m/f/d)](https://career.quantum-systems.com/o/ai-software-engineer-mfd)

---

## Project Structure

```text
efficient-vision-benchmark/
├── 01_run_benchmark.py
├── 02_generate_visualizations.py
├── 03_finetune_models.py
├── configs/
│   └── models.yaml
├── docs/
│   ├── index.html
│   ├── site_data.js
│   ├── site_data.json
│   └── assets/
├── results/
│   ├── benchmark_results.json
│   ├── figures/
│   └── finetune/
├── src/
│   ├── data.py
│   ├── metrics.py
│   ├── models.py
│   ├── timing.py
│   └── training.py
├── REPORT.md
└── requirements.txt
```

---

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. *Deep Residual Learning for Image Recognition.* CVPR 2016. [arXiv](https://arxiv.org/abs/1512.03385)
2. Howard, A., et al. *Searching for MobileNetV3.* ICCV 2019. [arXiv](https://arxiv.org/abs/1905.02244)
3. Tan, M. & Le, Q. *EfficientNet: Rethinking Model Scaling.* ICML 2019. [arXiv](https://arxiv.org/abs/1905.11946)
4. Liu, Z., et al. *A ConvNet for the 2020s.* CVPR 2022. [arXiv](https://arxiv.org/abs/2201.03545)
5. Mehta, S. & Rastegari, M. *Separable Self-attention for Mobile Vision Transformers.* TPAMI 2023. [arXiv](https://arxiv.org/abs/2206.02680)
6. Cai, H., et al. *EfficientViT.* ICCV 2023. [arXiv](https://arxiv.org/abs/2305.07027)
7. Wang, A., et al. *RepViT.* CVPR 2024. [arXiv](https://arxiv.org/abs/2307.09283)
8. Qin, D., et al. *MobileNetV4.* CVPR 2024. [arXiv](https://arxiv.org/abs/2404.10518)
9. Helber, P., et al. *EuroSAT.* IEEE JSTARS 2019. [IEEE](https://ieeexplore.ieee.org/document/8736785)

---

## License

MIT. See [LICENSE](LICENSE).
