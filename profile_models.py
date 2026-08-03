import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import numpy as np
import onnxruntime as ort


def profile_session(model_path, provider, input_name, dummy_input, iterations=1000, warmup=10):
    """Runs a controlled profiling loop for a specific execution provider."""
    print(f"  Targeting Backend: {provider}...")

    # 1. Initialize the session with an explicit execution provider
    try:
        session = ort.InferenceSession(model_path, providers=[provider])
        # Verify that the engine actually accepted the requested hardware
        if provider not in session.get_providers():
            print(f"  [WARNING] {provider} not supported by local environment. Falling back.")
            return None
    except Exception as e:
        print(f"  [ERROR] Failed to initialize {provider}: {e}")
        return None

    # 2. THE WARMUP PHASE
    # Hardware accelerators require initial cycles to allocate VRAM, cache graphs, and compile kernels
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_input})

    # 3. THE BENCHMARKING LOOP
    latencies = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        end_time = time.perf_counter()

        # Convert to milliseconds
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)

    # 4. Compute Statistical Metrics
    mean_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)

    print(f"    Mean Latency: {mean_latency:.4f} ms")
    print(f"    P95 Latency:  {p95_latency:.4f} ms")
    return {"mean": mean_latency, "p95": p95_latency}


def run_benchmark():
    print("==================================================")
    print("STARTING HARDWARE PERFORMANCE PROFILING")
    print("==================================================")

    # --- MODEL 1: ANOMALY DETECTOR ---
    print("\n[ MODEL 1: 1D ANOMALY DETECTOR ]")
    model1_path = "inference/anomaly_detector.onnx"
    dummy_input1 = np.random.randn(1, 64).astype(np.float32)

    profile_session(model1_path, 'CPUExecutionProvider', 'sensor_sequence', dummy_input1)
    profile_session(model1_path, 'CUDAExecutionProvider', 'sensor_sequence', dummy_input1)

    # --- MODEL 2: VISION CLASSIFIER ---
    print("\n[ MODEL 2: 2D VISION CLASSIFIER ]")
    model2_path = "inference/model2_cifar10.onnx"
    dummy_input2 = np.random.randn(1, 3, 32, 32).astype(np.float32)

    profile_session(model2_path, 'CPUExecutionProvider', 'input_image', dummy_input2)
    profile_session(model2_path, 'CUDAExecutionProvider', 'input_image', dummy_input2)


if __name__ == "__main__":
    run_benchmark()