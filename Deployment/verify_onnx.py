import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import onnx
import onnxruntime as ort
import numpy as np


def verify_vision_model():
    print("\n--- Verifying Vision Classifier ---")
    model_path = "../inference/model2_cifar10.onnx"

    # 1. Structural Check
    try:
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        print("STATUS: [PASS] Graph structure is valid and uncorrupted.")
    except Exception as e:
        print(f"STATUS: [FAIL] Structural corruption detected: {e}")
        return

    # 2. Execution Check
    # CRITICAL: ONNX runtime uses NumPy arrays, NOT PyTorch tensors!
    print("Booting C++ Inference Engine...")
    ort_session = ort.InferenceSession(model_path)

    # Generate dummy data matching our static (1, 3, 32, 32) shape
    # We must explicitly cast to float32 to match the C++ memory allocation
    dummy_input = np.random.randn(1, 3, 32, 32).astype(np.float32)

    # Run the engine
    # We pass 'None' to fetch all outputs, and map the input to the port name we defined
    outputs = ort_session.run(None, {"input_image": dummy_input})

    output_tensor = outputs[0]
    print(f"Engine Output Shape: {output_tensor.shape}")

    if output_tensor.shape == (1, 10):
        print("STATUS: [PASS] Vision model execution successful. Model is ready for edge deployment.")
    else:
        print("STATUS: [FAIL] Dimension mismatch during C++ execution.")


def verify_anomaly_model():
    print("\n--- Verifying Anomaly Detector ---")
    model_path = "../inference/anomaly_detector.onnx"

    # 1. Structural Check
    try:
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        print("STATUS: [PASS] Graph structure is valid and uncorrupted.")
    except Exception as e:
        print(f"STATUS: [FAIL] Structural corruption detected: {e}")
        return

    # 2. Execution Check
    print("Booting C++ Inference Engine...")
    ort_session = ort.InferenceSession(model_path)

    # Generate 1D dummy sensor sequence (1, 64)
    dummy_input = np.random.randn(1, 64).astype(np.float32)

    outputs = ort_session.run(None, {"sensor_sequence": dummy_input})

    output_tensor = outputs[0]
    print(f"Engine Output Shape: {output_tensor.shape}")

    if output_tensor.shape == (1, 64):
        print("STATUS: [PASS] Anomaly model execution successful. Model is ready for edge deployment.")
    else:
        print("STATUS: [FAIL] Dimension mismatch during C++ execution.")


if __name__ == "__main__":
    verify_vision_model()
    verify_anomaly_model()