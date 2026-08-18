import os
import numpy as np
from Deployment.inference_wrapper import EdgeInferenceEngine
from benchmarks.benchmark_logger import log_benchmark


def run_gpu_test():
    print("--- INITIATING GPU HARDWARE BENCHMARK ---")

    # Target the float32 model (GPUs actually prefer float32/float16 over int8!)
    model_path = os.path.join("../inference", "model2_cifar10.onnx")

    if not os.path.exists(model_path):
        print(f"[ERROR] Could not find {model_path}. Make sure it is compiled!")
        return

    # Initialize with the new GPU toggle set to True
    try:
        engine = EdgeInferenceEngine(model_path, use_gpu=True)
    except Exception as e:
        print("\n[CRITICAL ERROR] Failed to boot GPU. Check CUDA installation.")
        print(e)
        return

    input_shape = engine.session.get_inputs()[0].shape
    if not isinstance(input_shape[0], int):
        input_shape[0] = 1

    print(f"\nSimulating 1000 inferences on Graphics Processing Unit...")

    latencies = []

    # Execute the stress test
    for i in range(1000):
        dummy_data = np.random.randn(*input_shape).astype(np.float32)
        _, latency = engine.predict(dummy_data)
        latencies.append(latency)

    # Isolate the cold start
    cold_start = latencies[0]
    warm_latencies = latencies[1:]

    avg_latency = np.mean(warm_latencies)
    p99_latency = np.percentile(warm_latencies, 99)

    print("\n--- TANGIBLE ARTIFACT: GPU RESULTS ---")
    print(f"Cold Start Latency: {cold_start:.2f} ms")
    print(f"Average Latency:    {avg_latency:.4f} ms")
    print(f"99th Percentile:    {p99_latency:.4f} ms")
    print("----------------------------------------")

    # --- AUTO-LOGGER INTEGRATION ---
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    log_benchmark(
        model_id="Model 2 (GPU)",
        architecture="2D CNN",
        precision="float32",
        metric_name="Latency Test",
        metric_value=0.0,  # We are only tracking hardware speed here
        latency_ms=avg_latency,
        batch_size=1,
        model_size_mb=model_size_mb
    )


if __name__ == "__main__":
    run_gpu_test()