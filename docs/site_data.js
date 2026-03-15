window.BENCHMARK_SITE_DATA = {
  "generated_at": "2026-03-14T23:04:21.266696",
  "meta": {
    "timestamp": "2026-03-14T23:04:21.266696",
    "device": "cuda",
    "pytorch_version": "2.9.1+cu128",
    "cuda_available": true,
    "gpu_name": "NVIDIA GeForce RTX 3060 Laptop GPU",
    "dataset": "EuroSAT",
    "protocol": "head_reset_transfer_baseline",
    "protocol_label": "Head-reset transfer baseline",
    "protocol_description": "Each pretrained backbone is adapted to EuroSAT by replacing the final classifier with a fresh 10-class layer and evaluating before training.",
    "split_seed": 42,
    "split_counts": {
      "train": 18900,
      "val": 4050,
      "test": 4050
    },
    "class_names": [
      "AnnualCrop",
      "Forest",
      "HerbaceousVegetation",
      "Highway",
      "Industrial",
      "Pasture",
      "PermanentCrop",
      "Residential",
      "River",
      "SeaLake"
    ]
  },
  "summary": {
    "model_count": 9,
    "best_accuracy_model": {
      "name": "FastViT-MCI4",
      "value": 13.41,
      "unit": "%"
    },
    "fastest_cpu_model": {
      "name": "MobileNetV4-Small",
      "value": 15.62,
      "unit": "ms"
    },
    "best_size_efficiency_model": {
      "name": "MobileViTv2-0.5",
      "value": 6.75,
      "unit": "acc / M param"
    },
    "recommended_finetune_targets": [
      "mobilevitv2_050",
      "mobilenet_v3_large",
      "convnext_tiny"
    ],
    "best_finetuned_model": {
      "model_key": "convnext_tiny",
      "name": "ConvNeXt-Tiny",
      "top1_accuracy": 99.14,
      "best_val_accuracy": 98.99,
      "gain": 87.76
    }
  },
  "models": [
    {
      "experiment": "baseline",
      "experiment_label": "Head-reset transfer baseline",
      "model_key": "mobilenetv4_conv_small",
      "name": "MobileNetV4-Small",
      "year": 2024,
      "source": "timm",
      "top1_accuracy": 8.0,
      "top5_accuracy": 47.56,
      "cpu_latency_ms": 15.62,
      "gpu_latency_ms": 8.62,
      "params_m": 2.51,
      "size_mb": 9.56,
      "macs_m": 188.26,
      "throughput_bs1": 138.7,
      "throughput_bs16": 2498.3,
      "efficiency_acc_per_mparam": 3.187,
      "efficiency_acc_per_ms": 0.512
    },
    {
      "experiment": "baseline",
      "experiment_label": "Head-reset transfer baseline",
      "model_key": "mobilenet_v3_large",
      "name": "MobileNetV3-Large",
      "year": 2019,
      "source": "torchvision",
      "top1_accuracy": 10.07,
      "top5_accuracy": 55.7,
      "cpu_latency_ms": 19.87,
      "gpu_latency_ms": 9.62,
      "params_m": 4.21,
      "size_mb": 16.08,
      "macs_m": 230.05,
      "throughput_bs1": 97.7,
      "throughput_bs16": 1496.0,
      "efficiency_acc_per_mparam": 2.392,
      "efficiency_acc_per_ms": 0.507
    },
    {
      "experiment": "baseline",
      "experiment_label": "Head-reset transfer baseline",
      "model_key": "efficientnet_b0",
      "name": "EfficientNet-B0",
      "year": 2019,
      "source": "torchvision",
      "top1_accuracy": 5.75,
      "top5_accuracy": 42.22,
      "cpu_latency_ms": 28.02,
      "gpu_latency_ms": 12.84,
      "params_m": 4.02,
      "size_mb": 15.34,
      "macs_m": 408.93,
      "throughput_bs1": 75.8,
      "throughput_bs16": 960.1,
      "efficiency_acc_per_mparam": 1.43,
      "efficiency_acc_per_ms": 0.205
    },
    {
      "experiment": "baseline",
      "experiment_label": "Head-reset transfer baseline",
      "model_key": "mobilevitv2_050",
      "name": "MobileViTv2-0.5",
      "year": 2022,
      "source": "timm",
      "top1_accuracy": 7.56,
      "top5_accuracy": 44.74,
      "cpu_latency_ms": 30.39,
      "gpu_latency_ms": 14.98,
      "params_m": 1.12,
      "size_mb": 4.26,
      "macs_m": 471.64,
      "throughput_bs1": 81.7,
      "throughput_bs16": 1020.2,
      "efficiency_acc_per_mparam": 6.75,
      "efficiency_acc_per_ms": 0.249
    },
    {
      "experiment": "baseline",
      "experiment_label": "Head-reset transfer baseline",
      "model_key": "efficientvit_m0",
      "name": "EfficientViT-M0",
      "year": 2023,
      "source": "timm",
      "top1_accuracy": 11.09,
      "top5_accuracy": 47.48,
      "cpu_latency_ms": 31.88,
      "gpu_latency_ms": 29.19,
      "params_m": 2.16,
      "size_mb": 8.23,
      "macs_m": 79.29,
      "throughput_bs1": 41.2,
      "throughput_bs16": 594.7,
      "efficiency_acc_per_mparam": 5.134,
      "efficiency_acc_per_ms": 0.348
    },
    {
      "experiment": "baseline",
      "experiment_label": "Head-reset transfer baseline",
      "model_key": "repvit_m0_9",
      "name": "RepViT-M0.9",
      "year": 2024,
      "source": "timm",
      "top1_accuracy": 12.52,
      "top5_accuracy": 47.31,
      "cpu_latency_ms": 50.31,
      "gpu_latency_ms": 20.95,
      "params_m": 4.72,
      "size_mb": 18.01,
      "macs_m": 838.93,
      "throughput_bs1": 57.2,
      "throughput_bs16": 777.2,
      "efficiency_acc_per_mparam": 2.653,
      "efficiency_acc_per_ms": 0.249
    },
    {
      "experiment": "baseline",
      "experiment_label": "Head-reset transfer baseline",
      "model_key": "resnet50",
      "name": "ResNet-50",
      "year": 2016,
      "source": "torchvision",
      "top1_accuracy": 12.89,
      "top5_accuracy": 62.07,
      "cpu_latency_ms": 53.1,
      "gpu_latency_ms": 10.36,
      "params_m": 23.53,
      "size_mb": 89.75,
      "macs_m": 4130.41,
      "throughput_bs1": 114.6,
      "throughput_bs16": 423.2,
      "efficiency_acc_per_mparam": 0.548,
      "efficiency_acc_per_ms": 0.243
    },
    {
      "experiment": "baseline",
      "experiment_label": "Head-reset transfer baseline",
      "model_key": "convnext_tiny",
      "name": "ConvNeXt-Tiny",
      "year": 2022,
      "source": "torchvision",
      "top1_accuracy": 11.38,
      "top5_accuracy": 46.54,
      "cpu_latency_ms": 55.22,
      "gpu_latency_ms": 8.8,
      "params_m": 27.83,
      "size_mb": 106.15,
      "macs_m": 4487.59,
      "throughput_bs1": 101.7,
      "throughput_bs16": 327.6,
      "efficiency_acc_per_mparam": 0.409,
      "efficiency_acc_per_ms": 0.206
    },
    {
      "experiment": "baseline",
      "experiment_label": "Head-reset transfer baseline",
      "model_key": "fastvit_mci4",
      "name": "FastViT-MCI4",
      "year": 2025,
      "source": "timm",
      "top1_accuracy": 13.41,
      "top5_accuracy": 55.9,
      "cpu_latency_ms": 385.84,
      "gpu_latency_ms": 56.13,
      "params_m": 318.68,
      "size_mb": 1215.67,
      "macs_m": 22112.06,
      "throughput_bs1": 16.5,
      "throughput_bs16": 65.7,
      "efficiency_acc_per_mparam": 0.042,
      "efficiency_acc_per_ms": 0.035
    }
  ],
  "figures": {
    "accuracy_vs_latency": "assets/figures/accuracy_vs_latency.png",
    "parameter_footprint": "assets/figures/parameter_footprint.png",
    "latency_breakdown": "assets/figures/cpu_gpu_latency_breakdown.png",
    "finetune_accuracy_delta": "assets/figures/baseline_vs_finetuned_accuracy.png",
    "dataset_mosaic": "assets/figures/eurosat_testset_mosaic.png",
    "qualitative_before_after": "assets/figures/qualitative_before_after.png"
  },
  "finetune": {
    "available": true,
    "completed_models": [
      "convnext_tiny",
      "mobilenet_v3_large",
      "mobilevitv2_050"
    ],
    "leaderboard": [
      {
        "model_key": "convnext_tiny",
        "name": "ConvNeXt-Tiny",
        "top1_accuracy": 99.14,
        "best_val_accuracy": 98.99,
        "gain": 87.76
      },
      {
        "model_key": "mobilenet_v3_large",
        "name": "MobileNetV3-Large",
        "top1_accuracy": 98.74,
        "best_val_accuracy": 98.54,
        "gain": 88.67
      },
      {
        "model_key": "mobilevitv2_050",
        "name": "MobileViTv2-0.5",
        "top1_accuracy": 97.9,
        "best_val_accuracy": 98.22,
        "gain": 90.34
      }
    ],
    "recommended_export_model": "convnext_tiny",
    "accuracy_comparison": [
      {
        "model_key": "convnext_tiny",
        "name": "ConvNeXt-Tiny",
        "baseline_top1_accuracy": 11.38,
        "finetuned_top1_accuracy": 99.14,
        "gain": 87.76
      },
      {
        "model_key": "mobilenet_v3_large",
        "name": "MobileNetV3-Large",
        "baseline_top1_accuracy": 10.07,
        "finetuned_top1_accuracy": 98.74,
        "gain": 88.67
      },
      {
        "model_key": "mobilevitv2_050",
        "name": "MobileViTv2-0.5",
        "baseline_top1_accuracy": 7.56,
        "finetuned_top1_accuracy": 97.9,
        "gain": 90.34
      }
    ],
    "deployment_candidate": {
      "model_key": "mobilenet_v3_large",
      "name": "MobileNetV3-Large",
      "reason": "Fastest model within 1 percentage point of the top fine-tuned accuracy."
    }
  },
  "deployment": {
    "available": true,
    "selected_model": {
      "model_key": "mobilenet_v3_large",
      "name": "MobileNetV3-Large",
      "checkpoint": "results/finetune/mobilenet_v3_large/best.pt",
      "input_size": 224,
      "test_top1_accuracy": 98.74,
      "best_val_accuracy": 98.54,
      "checkpoint_epoch": 13,
      "family": "NAS-designed with squeeze-excite \u2014 mobile deployment standard"
    },
    "artifacts": {
      "onnx_fp32": {
        "path": "results/deployment/mobilenet_v3_large/mobilenet_v3_large_fp32.onnx",
        "opset": 19,
        "dynamo": false,
        "dynamic_batch": true,
        "size_mb": 16.07
      },
      "onnx_validation": {
        "max_abs_diff": 0.003182,
        "mean_abs_diff": 0.001391
      },
      "calibration_data": {
        "path": "results/deployment/mobilenet_v3_large/calibration_data.npz",
        "num_samples": 256,
        "class_names": [
          "AnnualCrop",
          "Forest",
          "HerbaceousVegetation",
          "Highway",
          "Industrial",
          "Pasture",
          "PermanentCrop",
          "Residential",
          "River",
          "SeaLake"
        ],
        "size_mb": 147.0
      },
      "onnx_int8": {
        "status": "ok",
        "path": "results/deployment/mobilenet_v3_large/mobilenet_v3_large_int8_qdq.onnx",
        "backend": "onnxruntime-static-qdq",
        "size_mb": 4.25
      }
    },
    "meta": {
      "generated_at": "2026-03-15T00:30:37",
      "device": "cuda",
      "elapsed_minutes": 0.35,
      "onnxruntime_available": true,
      "trtexec_available": false
    }
  },
  "qualitative": {
    "available": true,
    "model_name": "MobileNetV3-Large",
    "sample_count": 6,
    "example_classes": [
      "Annual Crop",
      "Highway",
      "Industrial",
      "Residential",
      "River",
      "Sea Lake"
    ],
    "summary": "Real held-out EuroSAT test imagery. The dataset panel shows one test tile per class, and the before/after panel uses MobileNetV3-Large to show how supervised adaptation corrects the fresh 10-class head on representative aerial samples."
  },
  "links": {
    "repo_url": "https://github.com/KenanTrivedi/efficient-vision-benchmark",
    "readme_url": "https://github.com/KenanTrivedi/efficient-vision-benchmark/blob/main/README.md",
    "pages_url": "https://kenantrivedi.github.io/efficient-vision-benchmark/",
    "pdf_path": "assets/EuroSAT_Edge_Vision_Benchmark_Portfolio.pdf"
  }
};
