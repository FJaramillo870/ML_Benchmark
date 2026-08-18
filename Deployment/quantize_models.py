import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from onnxruntime.quantization import quantize_dynamic, QuantType, shape_inference


def apply_quantization(input_model_path, output_model_path):
    print(f"--- QUANTIZING: {input_model_path} ---")

    if not os.path.exists(input_model_path):
        print(f"[ERROR] Could not find {input_model_path}.")
        return

    # 1. The Fix: Pre-process the model to clean up PyTorch shape anomalies
    preprocessed_path = input_model_path.replace(".onnx", "_preprocessed.onnx")
    print("Running ONNX Pre-processing (fixing shape inference issues)...")
    shape_inference.quant_pre_process(
        input_model_path=input_model_path,
        output_model_path=preprocessed_path,
        skip_optimization=False
    )

    # 2. Run the ONNX Dynamic Quantizer on the CLEANED model
    print("Running Dynamic Quantization...")
    quantize_dynamic(
        model_input=preprocessed_path,
        model_output=output_model_path,
        weight_type=QuantType.QInt8
    )

    # 3. Calculate and display the physical file size difference
    original_size_mb = os.path.getsize(input_model_path) / (1024 * 1024)
    quantized_size_mb = os.path.getsize(output_model_path) / (1024 * 1024)

    print(f"Original Size:  {original_size_mb:.2f} MB")
    print(f"Quantized Size: {quantized_size_mb:.2f} MB")
    print(f"Successfully saved to: {output_model_path}\n")


if __name__ == "__main__":
    print("Booting Edge AI Quantization Engine...\n")

    # Correctly pointing to the inference folder
    model2_input = os.path.join("inference", "model2_cifar10.onnx")
    model2_output = os.path.join("inference", "model2_cifar10_quantized.onnx")

    apply_quantization(model2_input, model2_output)