# Efficient Vision Model Benchmark

> A reproducible benchmarking pipeline for comparing classical, mobile-optimized, and modern vision architectures under identical evaluation conditions.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Motivation

Choosing the right vision backbone matters — whether for classification, feature extraction, or downstream fine-tuning. Published numbers from different papers use different training setups, augmentation strategies, and hardware, making direct comparisons unreliable.

This project benchmarks a curated set of architectures **under the same controlled conditions**: identical preprocessing, evaluation protocol, hardware, and measurement methodology. The goal is to produce a clear, honest comparison of **accuracy–latency–size trade-offs** that supports practical model selection.

## Models

The benchmark covers 8 architectures spanning classical CNNs (2016), mobile-optimized models (2019–2021), and modern designs (2021–2022):

| Model | Year | Params (M) | FLOPs (G) | ImageNet Top-1 (%) | Architecture Type |
|-------|------|-----------|-----------|---------------------|-------------------|
| ResNet-50 | 2016 | 25.6 | 4.09 | 80.86 | Residual baseline |
| MobileNetV3-Large | 2019 | 5.5 | 0.22 | 75.27 | NAS + squeeze-excite |
| EfficientNet-B0 | 2019 | 5.3 | 0.39 | 77.69 | Compound scaling |
| RegNetY-400MF | 2020 | 4.3 | 0.40 | 75.80 | Parametric design space |
| EfficientNetV2-S | 2021 | 21.5 | 2.87 | 84.23 | Fused-MBConv + progressive |
| Swin-Tiny | 2021 | 28.3 | 4.49 | 81.47 | Shifted-window transformer |
| ConvNeXt-Tiny | 2022 | 28.6 | 4.46 | 82.52 | Modernized pure CNN |
| MaxViT-Tiny | 2022 | 30.9 | 5.45 | 83.62 | Multi-axis attention |

*ImageNet Top-1 values from torchvision model zoo (official pretrained weights).*

## Evaluation Protocol

- **Dataset**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — 10,000 test images, 10 classes
  - Auto-downloaded via `torchvision.datasets.CIFAR10`
  - Resized to 224×224 (or model-native resolution) with ImageNet normalization
- **Accuracy**: Top-1 and Top-5 on CIFAR-10 test set (pretrained weights, adapted classifier)
- **Latency**: Mean ± std over 50 runs, after 10 warmup iterations, single-thread CPU (batch=1)
- **Model size**: Parameter count, disk footprint (MB), FLOPs via [ptflops](https://github.com/sovrasov/flops-counter.pytorch)
- **Throughput**: Images/second at batch sizes 1, 16, 64

## Quick Start

```bash
# Clone the repository
git clone https://github.com/KenanTrivedi/efficient-vision-benchmark.git
cd efficient-vision-benchmark

# Set up environment
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run the full benchmark (downloads CIFAR-10 automatically)
python benchmark.py

# Benchmark a single model
python benchmark.py --model convnext_tiny

# List all available models
python benchmark.py --list

# Generate comparison visualizations from results
python visualize.py
```

## Project Structure

```
efficient-vision-benchmark/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── benchmark.py              # Main entry point — runs full benchmark suite
├── visualize.py              # Generates comparison charts from results
├── configs/
│   └── models.yaml           # Model registry with paper references
├── src/
│   ├── __init__.py
│   ├── models.py             # Model loading, classifier adaptation
│   ├── data.py               # CIFAR-10 loading, transforms, dummy inputs
│   ├── timing.py             # Latency & throughput measurement
│   └── metrics.py            # Accuracy evaluation, FLOPs estimation
└── results/
    ├── benchmark_results.json
    └── figures/
        ├── accuracy_vs_latency.png
        ├── model_size_comparison.png
        └── tradeoff_radar.png
```

## Visualizations

After running `python benchmark.py && python visualize.py`:

1. **Accuracy vs. Latency** — scatter plot showing the Pareto front of best trade-offs
2. **Model Size Comparison** — parameter count and disk footprint side by side
3. **Trade-off Radar** — normalized multi-dimensional view (accuracy, speed, compactness, efficiency)

All figures are generated with matplotlib in a dark theme and saved to `results/figures/`.

## Roadmap

- [x] Core benchmark pipeline (accuracy, latency, model size, FLOPs)
- [x] CIFAR-10 evaluation with standardized preprocessing
- [x] 8-model comparison including modern architectures (ConvNeXt, Swin, MaxViT)
- [x] Visualization suite with dark-themed charts
- [ ] ONNX Runtime inference comparison
- [ ] INT8 post-training quantization benchmarks
- [ ] ImageNet-1k validation set evaluation
- [ ] Fine-tuning comparison (linear probe vs. full fine-tune)
- [ ] Latency profiling on NVIDIA Jetson Orin Nano

## References

| Paper | Authors | Venue |
|-------|---------|-------|
| [Deep Residual Learning](https://arxiv.org/abs/1512.03385) | He et al. | CVPR 2016 |
| [MobileNetV3](https://arxiv.org/abs/1905.02244) | Howard et al. | ICCV 2019 |
| [EfficientNet](https://arxiv.org/abs/1905.11946) | Tan & Le | ICML 2019 |
| [EfficientNetV2](https://arxiv.org/abs/2104.00298) | Tan & Le | ICML 2021 |
| [RegNet](https://arxiv.org/abs/2003.13678) | Radosavovic et al. | CVPR 2020 |
| [Swin Transformer](https://arxiv.org/abs/2103.14030) | Liu et al. | ICCV 2021 |
| [ConvNeXt](https://arxiv.org/abs/2201.03545) | Liu et al. | CVPR 2022 |
| [MaxViT](https://arxiv.org/abs/2204.01697) | Tu et al. | ECCV 2022 |

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- torchvision ≥ 0.15
- See [`requirements.txt`](requirements.txt) for full list

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Independent project focused on reproducibility, practical engineering, and honest model comparison.*
