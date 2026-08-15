import onnxruntime as ort
import numpy as np
import time
import os


class EdgeInferenceEngine:
    """
    A unified wrapper to load and execute ANY compiled ONNX model.
    This is the core engine that will eventually run on the Raspberry Pi.
    """

    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        print(f"[SYSTEM] Booting Edge Engine for: {os.path.basename(model_path)}")

        # 1. Initialize ONNX Runtime Session
        self.session = ort.InferenceSession(model_path)

        # 2. Automatically detect the input node name
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, input_data):
        """
        Takes raw input data, runs it through the hardware engine, and returns the output + latency.
        """
        # ONNX strictly requires NumPy arrays
        if not isinstance(input_data, np.ndarray):
            # Defaulting to float32 as that is standard for ONNX inputs, even for quantized models
            input_data = np.array(input_data, dtype=np.float32)

        # Execute hardware-accelerated inference
        start_time = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: input_data})
        latency_ms = (time.perf_counter() - start_time) * 1000

        return outputs[0], latency_ms


if __name__ == "__main__":
    print("--- TESTING UNIFIED INFERENCE ENGINE ---")

    # 1. Test it with your quantized Vision Model
    vision_model_path = os.path.join("inference", "model2_cifar10_quantized.onnx")
    engine = EdgeInferenceEngine(vision_model_path)

    # Create a dummy image (Batch Size 1, 3 Channels, 32x32 pixels)
    dummy_image = np.random.randn(1, 3, 32, 32).astype(np.float32)

    # Run unified prediction
    prediction, latency = engine.predict(dummy_image)

    print("\n--- TANGIBLE ARTIFACT ---")
    print(f"Prediction Shape: {prediction.shape} (Expected 1 batch, 10 classes)")
    print(f"Inference Time:   {latency:.4f} ms")
    print("STATUS: [OK] Unified API successfully executed model.")