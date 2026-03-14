import os
import torch
import json
from src.models import load_model_config, get_model, get_model_info
from src.data import get_eurosat_loaders
from src.timing import measure_latency, measure_throughput
from src.metrics import evaluate_accuracy, estimate_flops
from datetime import datetime

RESULTS_DIR = "results"
RESULTS_FILE = "results/benchmark_results.json"

def run_all():
    config = load_model_config()
    model_keys = list(config.keys())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading data...")
    _, test_loader = get_eurosat_loaders(input_size=224, batch_size=64, num_workers=2)
    all_results = {}
    
    for i, key in enumerate(model_keys, 1):
        try:
            print(f"\n[{i}/{len(model_keys)}] Benchmarking {key}...")
            model = get_model(key, num_classes=10)
            in_size = config[key].get('input_size', 224)
            model_info = get_model_info(model)
            flops_info = estimate_flops(model, in_size)
            print("CPU Latency...")
            lat_cpu = measure_latency(model, in_size, "cpu", 5, 20, 1)
            lat_gpu = None
            if device == "cuda":
                print("GPU Latency...")
                lat_gpu = measure_latency(model, in_size, "cuda", 5, 30, 1)
            print("Throughput...")
            throughput = measure_throughput(model, in_size, device, [1, 16], 1.0)
            print("Accuracy...")
            acc = evaluate_accuracy(model, test_loader, device)
            print(f"-> {acc['top1_accuracy']}%")
            
            all_results[key] = {
                "model_key": key,
                "model_info": model_info,
                "flops": flops_info,
                "latency_cpu": lat_cpu,
                "latency_gpu": lat_gpu,
                "throughput": {str(k): v for k, v in throughput.items()},
                "accuracy": acc,
            }
        except Exception as e:
            print(f"Error on {key}: {e}")
            
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "models": all_results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    
if __name__ == '__main__':
    run_all()
