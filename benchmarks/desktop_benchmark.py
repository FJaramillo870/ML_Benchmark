import onnxruntime as ort
import numpy as np
import time


def benchmark_desktop_model(model_path, input_shape, provider_list, hardware_name):
    print(f"\n--- Evaluating: {model_path} on {hardware_name} ---")
    try:
        # Initialize session with specific hardware provider
        session = ort.InferenceSession(model_path, providers=provider_list)

        # Get input name dynamically from the model
        input_name = session.get_inputs()[0].name

        # Generate dummy data
        dummy_data = np.random.randn(*input_shape).astype(np.float32)

        # Warmup phase
        print(f"Warming up {hardware_name}...")
        for _ in range(5):
            session.run(None, {input_name: dummy_data})

        # Benchmark phase
        print(f"Executing 50 {hardware_name} inference cycles...")
        latencies = []
        for _ in range(50):
            start_time = time.perf_counter()
            session.run(None, {input_name: dummy_data})
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

        avg_latency = sum(latencies) / len(latencies)
        print("STATUS: [SUCCESS]")
        print(f"Average {hardware_name} Latency: {avg_latency:.4f} ms")

    except Exception as e:
        print(f"STATUS: [FAILED] {e}")


if __name__ == "__main__":
    print("BOOTING DESKTOP HARDWARE BENCHMARK SUITE...")

    # 3x3 Matrix: Desktop Platform Execution
    models = [
        ("inference/anomaly_detector.onnx", (1, 64)),  # Model 1
        ("inference/model2_cifar10.onnx", (1, 3, 32, 32)),  # Model 2
        ("inference/model3_lstm.onnx", (1, 10, 1))  # Model 3
    ]

    for model_path, shape in models:
        # 1. Desktop CPU Test
        benchmark_desktop_model(
            model_path,
            shape,
            ['CPUExecutionProvider'],
            "Desktop CPU"
        )

        # 2. Desktop GPU Test
        benchmark_desktop_model(
            model_path,
            shape,
            ['CUDAExecutionProvider', 'CPUExecutionProvider'],
            "Desktop GPU"
        )