from Deployment.inference_wrapper import EdgeInferenceEngine
import numpy as np


def evaluate_base_model(model_name, input_shape):
    print(f"\n--- Evaluating: {model_name} ---")
    try:
        engine = EdgeInferenceEngine(model_name)

        # Generate dummy data matching the required input shape
        dummy_data = np.random.randn(*input_shape).astype(np.float32)

        # Warmup phase to stabilize hardware state
        print("Warming up hardware...")
        for _ in range(5):
            engine.predict(dummy_data)

        # Benchmark phase
        print("Executing 50 hardware inference cycles...")
        latencies = []
        for _ in range(50):
            _, latency = engine.predict(dummy_data)
            latencies.append(latency)

        avg_latency = sum(latencies) / len(latencies)
        print("STATUS: [SUCCESS]")
        print(f"Average ARM Latency: {avg_latency:.4f} ms")

    except Exception as e:
        print(f"STATUS: [FAILED] {e}")


if __name__ == "__main__":
    print("BOOTING EDGE HARDWARE BENCHMARK SUITE...")

    # 3x3 Matrix: Edge Platform Execution
    # Model 1: Anomaly Detector (MLP Autoencoder)
    evaluate_base_model("anomaly_detector.onnx", (1, 64))

    # Model 2: Vision Classifier (CNN)
    evaluate_base_model("model2_cifar10.onnx", (1, 3, 32, 32))

    # Model 3: Forecaster (LSTM)
    evaluate_base_model("model3_lstm.onnx", (1, 10, 1))