import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch


def demonstrate_quantization_memory():
    print("--- SIMULATING QUANTIZATION MEMORY SAVINGS ---")

    # 1. Create a dummy weight matrix (e.g., a layer in your CNN)
    # 10,000 x 10,000 matrix = 100 million weights
    print("Generating 100 million weights...")
    weights_float32 = torch.randn(10000, 10000, dtype=torch.float32)

    # 2. Simulate Quantization (converting to 8-bit integer)
    # We multiply by 100 and round to preserve a tiny bit of the decimal info as a whole number
    weights_int8 = (weights_float32 * 100).to(torch.int8)

    # 3. Calculate physical memory size in Megabytes (MB)
    # Formula: (bytes per element * total elements) / 1024 / 1024
    float32_size_mb = (weights_float32.element_size() * weights_float32.nelement()) / (1024 * 1024)
    int8_size_mb = (weights_int8.element_size() * weights_int8.nelement()) / (1024 * 1024)

    print("\n--- TANGIBLE ARTIFACT ---")
    print(f"Original float32 Memory: {float32_size_mb:.2f} MB")
    print(f"Quantized int8 Memory:   {int8_size_mb:.2f} MB")
    print("---------------------------------------------")
    print("Hardware impact: The model is physically 4x smaller and requires less RAM to boot.")


if __name__ == "__main__":
    demonstrate_quantization_memory()