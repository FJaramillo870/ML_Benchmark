import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from models.model2 import CIFAR10Classifier
from models.mlp_encoder import MLPAutoencoder
from models.forecasting_model import LSTMForecaster


def export_vision_model():
    print("\nBooting ONNX Exporter for Vision Classifier...")
    model = CIFAR10Classifier()
    model.load_state_dict(torch.load("../models/checkpoints/model2_cifar10.pth", weights_only=True))
    model.eval()

    # Static batch size of 1
    dummy_input = torch.randn(1, 3, 32, 32)

    # Routed to the correct folder
    onnx_file_path = "../inference/model2_cifar10.onnx"

    torch.onnx.export(
        model, dummy_input, onnx_file_path,
        export_params=True,
        opset_version=14,  # <-- DOWNGRADED TO 14 FOR EDGE
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['class_logits']
    )
    print(f"STATUS: [SUCCESS] Vision Model exported to {onnx_file_path}")


def export_anomaly_model():
    print("\nBooting ONNX Exporter for Anomaly Detector...")
    model = MLPAutoencoder()
    model.load_state_dict(torch.load("../models/checkpoints/trained_autoencoder.pth", weights_only=True))
    model.eval()

    # Static batch size of 1
    dummy_input = torch.randn(1, 64)
    onnx_file_path = "../inference/anomaly_detector.onnx"

    torch.onnx.export(
        model, dummy_input, onnx_file_path,
        export_params=True,
        opset_version=14,  # <-- DOWNGRADED TO 14 FOR EDGE
        do_constant_folding=True,
        input_names=['sensor_sequence'],
        output_names=['reconstructed_sequence']
    )
    print(f"STATUS: [SUCCESS] Anomaly Model exported to {onnx_file_path}")


def export_forecasting_model():
    print("\nBooting ONNX Exporter for LSTM Forecasting...")

    # NOTE: Add init parameters inside the parentheses if your class requires them!
    model = LSTMForecaster()

    model.load_state_dict(torch.load("../models/checkpoints/model3_lstm.pth", weights_only=True))
    model.eval()

    # Create dummy input (Adjust shape: Batch Size, Sequence Length, Features)
    dummy_input = torch.randn(1, 10, 1)
    onnx_file_path = "../inference/model3_lstm.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=18,  # Let PyTorch use its required version for LSTMs
        do_constant_folding=True,
        input_names=['sequence_input'],
        output_names=['forecast_output']
    )
    print(f"STATUS: [SUCCESS] LSTM Model exported to {onnx_file_path}")


if __name__ == "__main__":
    export_vision_model()
    export_anomaly_model()
    export_forecasting_model()