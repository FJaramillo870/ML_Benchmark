import os
import numpy as np
import psutil
from Deployment.inference_wrapper import EdgeInferenceEngine


def run_profiler(model_path, iterations=1000):
    print("--- STARTING HARDWARE PROFILING ---")

    # 1. Boot the engine
    engine = EdgeInferenceEngine(model_path)
    input_shape = engine.session.get_inputs()[0].shape
    if not isinstance(input_shape[0], int):
        input_shape[0] = 1

    print(f"Simulating {iterations} rapid-fire inferences...")
    print("Tracking latency and physical RAM usage...\n")

    # 2. Setup Tracking Variables
    latencies = []
    memory_footprints = []
    process = psutil.Process(os.getpid())  # Hooks into your OS to track this specific Python script

    # 3. The Stress Test Loop
    for _ in range(iterations):
        # Generate dummy sensor data
        dummy_data = np.random.randn(*input_shape).astype(np.float32)

        # Execute prediction
        _, latency = engine.predict(dummy_data)

        # Track physical memory (RAM) used by the system right now
        # RSS (Resident Set Size) is the true physical memory footprint
        current_ram_mb = process.memory_info().rss / (1024 * 1024)

        latencies.append(latency)
        memory_footprints.append(current_ram_mb)

    # 4. Data Analysis (Dropping the cold start for accurate averages)
    warm_latencies = latencies[1:]

    avg_latency = np.mean(warm_latencies)
    p99_latency = np.percentile(warm_latencies, 99)  # The worst-case latency (99th percentile)
    peak_ram = max(memory_footprints)

    print("--- TANGIBLE ARTIFACT: PROFILING RESULTS ---")
    print(f"Average Latency: {avg_latency:.4f} ms")
    print(f"99th Percentile: {p99_latency:.4f} ms (Worst-case speed)")
    print(f"Peak RAM Usage:  {peak_ram:.2f} MB")
    print("--------------------------------------------")
    print("STATUS: [OK] Hardware profiling complete and reliable.")


if __name__ == "__main__":
    # Test the profiler using the quantized model
    test_model = os.path.join("../inference", "model2_cifar10_quantized.onnx")
    run_profiler(test_model, iterations=1000)