window.BENCHMARK_SITE_DATA = {
  "generated_at": "2026-03-14T19:07:32.426855",
  "meta": {
    "timestamp": "2026-03-14T19:07:32.426855",
    "device": "cuda",
    "dataset": "EuroSAT",
    "protocol": "head_reset_transfer_baseline",
    "protocol_label": "Head-reset transfer baseline",
    "protocol_description": "Each pretrained backbone is adapted to EuroSAT by replacing the final classifier with a fresh 10-class layer and evaluating before training."
  },
  "summary": {
    "model_count": 8,
    "best_accuracy_model": {
      "name": "ResNet-50",
      "value": 12.98,
      "unit": "%"
    },
    "fastest_cpu_model": {
      "name": "MobileNetV3-Large",
      "value": 31.32,
      "unit": "ms"
    },
    "best_size_efficiency_model": {
      "name": "MobileViTv2-0.5",
      "value": 10.473,
      "unit": "acc / M param"
    },
    "recommended_finetune_targets": [
      "mobilevitv2_050",
      "mobilenet_v3_large",
      "convnext_tiny"
    ]
  },
  "models": [
    {
      "experiment": "baseline",
      "experiment_label": "Head-reset transfer baseline",
      "model_key": "mobilenet_v3_large",
      "name": "MobileNetV3-Large",
      "year": 2019,
      "source": "torchvision",
      "top1_accuracy": 9.94,
      "top5_accuracy": 49.23,
      "cpu_latency_ms": 31.32,
      "gpu_latency_ms": 7.71,
      "params_m": 4.21,
      "size_mb": 16.08,
      "macs_m": 230.05,
      "throughput_bs1": 131.7,
      "throughput_bs16": 1669.5,
      "efficiency_acc_per_mparam": 2.361,
      "efficiency_acc_per_ms": 0.317
    },
    {
      "experiment": "baseline",
      "experiment_label": "Head-reset transfer baseline",
      "model_key": "mobilenetv4_conv_small",
      "name": "MobileNetV4-Small",
      "year": 2024,
      "source": "timm",
      "top1_accuracy": 6.89,
      "top5_accuracy": 48.2,
      "cpu_latency_ms": 32.46,
      "gpu_latency_ms": 7.0,
      "params_m": 2.51,
      "size_mb": 9.56,
      "macs_m": 188.26,
      "throughput_bs1": 108.7,
      "throughput_bs16": 2226.7,
      "efficiency_acc_per_mparam": 2.745,
      "efficiency_acc_per_ms": 0.212
    },
    {
      "experiment": "baseline",
      "experiment_label": "Head-reset transfer baseline",
      "model_key": "efficientnet_b0",
      "name": "EfficientNet-B0",
      "year": 2019,
      "source": "torchvision",
      "top1_accuracy": 5.61,
      "top5_accuracy": 43.43,
      "cpu_latency_ms": 40.26,
      "gpu_latency_ms": 9.2,
      "params_m": 4.02,
      "size_mb": 15.34,
      "macs_m": 408.93,
      "throughput_bs1": 105.6,
      "throughput_bs16": 1009.7,
      "efficiency_acc_per_mparam": 1.396,
      "efficiency_acc_per_ms": 0.139
    },
    {
      "experiment": "baseline",
      "experiment_label": "Head-reset transfer baseline",
      "model_key": "mobilevitv2_050",
      "name": "MobileViTv2-0.5",
      "year": 2022,
      "source": "timm",
      "top1_accuracy": 11.73,
      "top5_accuracy": 51.59,
      "cpu_latency_ms": 46.72,
      "gpu_latency_ms": 12.69,
      "params_m": 1.12,
      "size_mb": 4.26,
      "macs_m": 471.64,
      "throughput_bs1": 74.7,
      "throughput_bs16": 1054.0,
      "efficiency_acc_per_mparam": 10.473,
      "efficiency_acc_per_ms": 0.251
    },
    {
      "experiment": "baseline",
      "experiment_label": "Head-reset transfer baseline",
      "model_key": "convnext_tiny",
      "name": "ConvNeXt-Tiny",
      "year": 2022,
      "source": "torchvision",
      "top1_accuracy": 12.28,
      "top5_accuracy": 50.03,
      "cpu_latency_ms": 58.32,
      "gpu_latency_ms": 6.33,
      "params_m": 27.83,
      "size_mb": 106.15,
      "macs_m": 4487.59,
      "throughput_bs1": 162.5,
      "throughput_bs16": 304.2,
      "efficiency_acc_per_mparam": 0.441,
      "efficiency_acc_per_ms": 0.211
    },
    {
      "experiment": "baseline",
      "experiment_label": "Head-reset transfer baseline",
      "model_key": "efficientvit_m0",
      "name": "EfficientViT-M0",
      "year": 2023,
      "source": "timm",
      "top1_accuracy": 8.86,
      "top5_accuracy": 56.95,
      "cpu_latency_ms": 73.27,
      "gpu_latency_ms": 40.45,
      "params_m": 2.16,
      "size_mb": 8.23,
      "macs_m": 79.29,
      "throughput_bs1": 25.4,
      "throughput_bs16": 380.5,
      "efficiency_acc_per_mparam": 4.102,
      "efficiency_acc_per_ms": 0.121
    },
    {
      "experiment": "baseline",
      "experiment_label": "Head-reset transfer baseline",
      "model_key": "resnet50",
      "name": "ResNet-50",
      "year": 2016,
      "source": "torchvision",
      "top1_accuracy": 12.98,
      "top5_accuracy": 52.75,
      "cpu_latency_ms": 85.38,
      "gpu_latency_ms": 17.2,
      "params_m": 23.53,
      "size_mb": 89.75,
      "macs_m": 4130.41,
      "throughput_bs1": 78.9,
      "throughput_bs16": 438.7,
      "efficiency_acc_per_mparam": 0.552,
      "efficiency_acc_per_ms": 0.152
    },
    {
      "experiment": "baseline",
      "experiment_label": "Head-reset transfer baseline",
      "model_key": "fastvit_mci4",
      "name": "FastViT-MCI4",
      "year": 2025,
      "source": "timm",
      "top1_accuracy": 3.36,
      "top5_accuracy": 42.03,
      "cpu_latency_ms": 898.06,
      "gpu_latency_ms": 80.4,
      "params_m": 318.68,
      "size_mb": 1215.67,
      "macs_m": 22112.06,
      "throughput_bs1": 12.2,
      "throughput_bs16": 76.1,
      "efficiency_acc_per_mparam": 0.011,
      "efficiency_acc_per_ms": 0.004
    }
  ],
  "figures": {
    "accuracy_vs_latency": "assets/figures/accuracy_vs_latency.png",
    "parameter_footprint": "assets/figures/parameter_footprint.png",
    "latency_breakdown": "assets/figures/cpu_gpu_latency_breakdown.png"
  },
  "finetune": {
    "available": false,
    "completed_models": []
  },
  "job_alignment": {
    "role_url": "https://career.quantum-systems.com/o/ai-software-engineer-mfd",
    "highlights": [
      "Preparing and selecting data, training and validating models, and deploying them on embedded UxV platforms.",
      "Experience with quantization, pruning, and distillation for resource-constrained environments.",
      "Hands-on deployment on NVIDIA Jetson, ARM, FPGA, or custom SoC edge hardware."
    ]
  }
};
