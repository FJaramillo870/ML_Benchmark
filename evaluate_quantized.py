import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import onnxruntime as ort
from benchmark_logger import log_benchmark

def evaluate_quantized_model():
    print("--- EVALUATING QUANTIZED MODEL 2 (VISION) ---")

    # 1. Point ONNX Runtime at your new int8 model
    model_path = os.path.join("inference", "model2_cifar10_quantized.onnx")
    if not os.path.exists(model_path):
        print(f"[ERROR] Could not find {model_path}")
        return

    # Boot up the lightweight ONNX inference engine
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    # 2. Setup CIFAR-10 Test Data
    # Notice we are forcing Batch Size = 1 to simulate a live edge camera feed!
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    correct = 0
    total = 0
    latencies = []

    print(f"Running inference on {len(testset)} images... (This might take a minute)")

    # 3. The Edge Inference Loop
    for images, labels in testloader:
        # ONNX doesn't understand PyTorch Tensors, so we convert to raw NumPy arrays
        numpy_images = images.numpy()

        start_time = time.perf_counter()

        # Execute the int8 model
        outputs = session.run(None, {input_name: numpy_images})

        end_time = time.perf_counter()

        # Determine which class (0-9) the model predicted
        predicted = np.argmax(outputs[0], axis=1)

        total += labels.size(0)
        correct += (predicted == labels.numpy()).sum().item()

        # Record latency in milliseconds
        latencies.append((end_time - start_time) * 1000)

    # 4. Calculate Final Metrics
    accuracy = 100 * correct / total
    avg_latency = sum(latencies) / len(latencies)

    print("\n--- TANGIBLE ARTIFACT ---")
    print(f"Total Correct:       {correct} / {total}")
    print(f"Quantized Accuracy:  {accuracy:.2f}%")
    print(f"Quantized Latency:   {avg_latency:.4f} ms (Batch Size 1)")

    # Calculate file size in MB
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    # Automatically log the run to the central CSV file
    log_benchmark(
        model_id="Model 2",
        architecture="2D CNN",
        precision="int8",
        metric_name="Accuracy (%)",
        metric_value=accuracy,
        latency_ms=avg_latency,
        batch_size=1,
        model_size_mb=model_size_mb
    )


if __name__ == "__main__":
    evaluate_quantized_model()