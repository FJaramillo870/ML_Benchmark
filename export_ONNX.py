import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from models.model2 import CIFAR10Classifier
from models.mlp_encoder import MLPAutoencoder


def export_vision_model():
    print("\nBooting ONNX Exporter for Vision Classifier...")
    model = CIFAR10Classifier()
    model.load_state_dict(torch.load("model2_cifar10.pth", weights_only=True))
    model.eval()

    # Static batch size of 1
    dummy_input = torch.randn(1, 3, 32, 32)

    # Routed to the correct folder
    onnx_file_path = "inference/model2_cifar10.onnx"

    torch.onnx.export(
        model, dummy_input, onnx_file_path,
        export_params=True,
        opset_version=18,  # Upgraded to satisfy Dynamo
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['class_logits']
        # dynamic_axes removed for strict hardware optimization
    )
    print(f"STATUS: [SUCCESS] Vision Model exported to {onnx_file_path}")


def export_anomaly_model():
    print("\nBooting ONNX Exporter for Anomaly Detector...")
    model = MLPAutoencoder()
    model.load_state_dict(torch.load("trained_autoencoder.pth", weights_only=True))
    model.eval()

    # Static batch size of 1
    dummy_input = torch.randn(1, 64)
    onnx_file_path = "inference/anomaly_detector.onnx"

    torch.onnx.export(
        model, dummy_input, onnx_file_path,
        export_params=True,
        opset_version=18,  # Upgraded to satisfy Dynamo
        do_constant_folding=True,
        input_names=['sensor_sequence'],
        output_names=['reconstructed_sequence']
        # dynamic_axes removed for strict hardware optimization
    )
    print(f"STATUS: [SUCCESS] Anomaly Model exported to {onnx_file_path}")


if __name__ == "__main__":
    export_vision_model()
    export_anomaly_model()