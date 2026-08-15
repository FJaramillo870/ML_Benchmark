import time
import numpy as np
import os
from inference_wrapper import EdgeInferenceEngine


def run_streaming_inference(model_path, iterations=50, target_hz=5):
    """
    Simulates a continuous hardware sensor feed (like a camera or temp sensor).
    target_hz: How many times per second the sensor takes a reading.
    """
    print("--- STARTING EDGE SENSOR STREAM ---")
    print(f"Target Polling Rate: {target_hz} Hz ({target_hz} reads per second)\n")

    # 1. Boot up the engine using your wrapper
    engine = EdgeInferenceEngine(model_path)

    # 2. Dynamically figure out what shape of data this specific model needs
    input_shape = engine.session.get_inputs()[0].shape
    # If ONNX says the batch size is dynamic (represented as a string or None), force it to 1
    if not isinstance(input_shape[0], int):
        input_shape[0] = 1

    print(f"[SYSTEM] Hardware sensor calibrated to shape: {input_shape}")
    print("Beginning stream... (Press Ctrl+C to stop manually)\n")

    try:
        # 3. The Continuous Edge Loop
        for i in range(1, iterations + 1):
            # A) Simulate the physical sensor grabbing new data
            live_sensor_data = np.random.randn(*input_shape).astype(np.float32)

            # B) Pass the live data directly into our AI engine
            prediction, latency = engine.predict(live_sensor_data)

            # C) Extract the system's "decision" (e.g., which class it detected)
            predicted_class = np.argmax(prediction[0])

            # D) Print the live telemetry
            print(f"[Reading {i:03d}] Latency: {latency:>5.2f} ms | System Decision: Class {predicted_class}")

            # E) Sleep to maintain our target polling rate (target_hz)
            time.sleep(1.0 / target_hz)

    except KeyboardInterrupt:
        print("\n[SYSTEM] Stream interrupted by operator.")

    print("\n--- STREAM TERMINATED ---")


if __name__ == "__main__":
    # We will test the stream using your highly optimized quantized model
    test_model = os.path.join("inference", "model2_cifar10_quantized.onnx")

    # Run the continuous sensor simulation for 50 readings at 5 readings per second
    run_streaming_inference(test_model, iterations=50, target_hz=5)