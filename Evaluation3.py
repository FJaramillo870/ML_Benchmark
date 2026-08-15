import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import os
import time

# Native imports work perfectly from the root directory!
from data.data_loader import get_energy_dataloaders
from models.forecasting_model import LSTMForecaster
from benchmark_logger import log_benchmark


def evaluate():
    print("--- EVALUATING MODEL 3 (LSTM) ---")

    # 1. Load the Test Data
    _, test_loader = get_energy_dataloaders(lookback_window=24, batch_size=32)

    # 2. Initialize and load the trained weights
    model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=1, output_size=1)

    # Clean, direct path to the model file
    model_path = os.path.join("models", "model3_lstm.pth")
    model.load_state_dict(torch.load(model_path))

    # Set to evaluation mode
    model.eval()

    criterion = nn.MSELoss()
    test_loss = 0.0
    latencies = []

    # 3. Run the Test Loop
    with torch.no_grad():
        for inputs, targets in test_loader:
            start_time = time.perf_counter()
            predictions = model(inputs)
            end_time = time.perf_counter()

            loss = criterion(predictions, targets)
            test_loss += loss.item()

            latencies.append((end_time - start_time) * 1000)

    avg_loss = test_loss / len(test_loader)
    avg_latency = sum(latencies) / len(latencies)

    print("\n--- TANGIBLE ARTIFACT ---")
    print(f"Test MSE Loss:       {avg_loss:.6f}")
    print(f"Avg CPU Latency:     {avg_latency:.2f} ms (Batch Size 32)")

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    log_benchmark(
        model_id="Model 3",
        architecture="LSTM",
        precision="float32",
        metric_name="Test MSE",
        metric_value=avg_loss,
        latency_ms=avg_latency,
        batch_size=32,
        model_size_mb=model_size_mb
    )


if __name__ == "__main__":
    evaluate()