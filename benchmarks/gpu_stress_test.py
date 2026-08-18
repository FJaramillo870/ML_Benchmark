import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import time
from models.model2 import CIFAR10Classifier

def run_stress_test():
    print("BOOTING HARDWARE STRESS TEST...")

    # 1. Setup Model and Massive Batch
    batch_size = 4096
    print(f"Generating massive payload: {batch_size} images simultaneously...")
    model = CIFAR10Classifier()
    model.eval()

    # 2. CPU Race
    print("\n--- CPU Execution ---")
    dummy_cpu = torch.randn(batch_size, 3, 32, 32)

    start_cpu = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy_cpu)
    end_cpu = time.perf_counter()
    print(f"CPU Time for {batch_size} images: {(end_cpu - start_cpu):.4f} seconds")

    # 3. GPU Race
    print("\n--- GPU Execution ---")
    if torch.cuda.is_available():
        # Move model and memory payload to the graphics card
        model = model.cuda()
        dummy_gpu = dummy_cpu.cuda()

        # Warmup GPU (Bypasses initial CUDA memory allocation overhead)
        with torch.no_grad():
            _ = model(dummy_gpu)
        torch.cuda.synchronize()

        # Benchmark
        start_gpu = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_gpu)
        torch.cuda.synchronize()  # Force Python to wait until CUDA finishes the math
        end_gpu = time.perf_counter()

        print(f"GPU Time for {batch_size} images: {(end_gpu - start_gpu):.4f} seconds")
        print(f"\nSTATUS: GPU is {(end_cpu - start_cpu) / (end_gpu - start_gpu):.1f}x faster at high throughput!")
    else:
        print("STATUS: [FAILED] CUDA is not available.")


if __name__ == "__main__":
    run_stress_test()