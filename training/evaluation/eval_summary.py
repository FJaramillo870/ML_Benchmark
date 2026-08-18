import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from benchmarks.benchmark_logger import log_benchmark

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

# 1. Import your blueprint and data
from models.mlp_encoder import MLPAutoencoder  # Swapped to your upgraded CNN!
from data.synthetic_power import SyntheticPowerDataset


def run_evaluation_summary():
    print("--- BOOTING EDGE AI EVALUATION SUMMARY ---")

    # 2. Load the Best Checkpoint
    model = MLPAutoencoder()
    # Update this to whatever you named your final 15-kernel save file
    model.load_state_dict(torch.load("../../Trash/frozen.pth", weights_only=True))
    model.eval()

    criterion = nn.MSELoss()
    anomaly_threshold = 0.1

    # 3. Generate the Dataset
    dataset = SyntheticPowerDataset(num_samples=1000, sequence_length=64)

    normal_mse_list = []
    anomaly_mse_list = []
    false_positives = 0
    false_negatives = 0

    print("Running 100 Normal Simulations...")
    print("Running 100 Motor Stall Simulations...")

    with torch.no_grad():
        # --- BATCH TEST: 100 WAVES ---
        for i in range(100):
            # Grab a normal sequence and add Batch dimension
            normal_wave = dataset[i].clone().unsqueeze(0)

            # Create an anomalous copy
            anomaly_wave = dataset[i].clone()

            # Randomize where the stall happens to prove the sliding window works!
            start_idx = random.randint(5, 55)
            anomaly_wave[start_idx:start_idx + 5] = anomaly_wave[start_idx:start_idx + 5] + 5.0
            anomaly_wave = anomaly_wave.unsqueeze(0)

            # Calculate Normal MSE
            normal_recon = model(normal_wave)
            normal_loss = criterion(normal_recon, normal_wave).item()
            normal_mse_list.append(normal_loss)
            if normal_loss > anomaly_threshold:
                false_positives += 1

            # Calculate Anomaly MSE
            anomaly_recon = model(anomaly_wave)
            anomaly_loss = criterion(anomaly_recon, anomaly_wave).item()
            anomaly_mse_list.append(anomaly_loss)
            if anomaly_loss < anomaly_threshold:
                false_negatives += 1

    # --- GENERATE LAB REPORT STATISTICS ---
    avg_normal = sum(normal_mse_list) / len(normal_mse_list)
    avg_anomaly = sum(anomaly_mse_list) / len(anomaly_mse_list)

    print("\n[ STATISTICS ]")
    print(f"Average Normal MSE:  {avg_normal:.4f}")
    print(f"Average Anomaly MSE: {avg_anomaly:.4f}")
    print(f"Signal Margin:       {(avg_anomaly / avg_normal):.1f}x")

    print("\n[ DEPLOYMENT METRICS ]")
    print(f"Threshold Set At: {anomaly_threshold}")
    print(f"False Positives:  {false_positives} / 100")
    print(f"False Negatives:  {false_negatives} / 100")

    # --- GENERATE MATPLOTLIB HISTOGRAM ---
    print("\nGenerating Histogram -> 'evaluation_histogram.png'")

    plt.figure(figsize=(10, 6))

    # Plot the two data clusters
    plt.hist(normal_mse_list, bins=50, range=(0.0, 1.0), alpha=0.7, color='green', label='Normal Operations')
    plt.hist(anomaly_mse_list, bins=50, range=(0.0, 1.0), alpha=0.7, color='red', label='Motor Stalls')

    # Draw the Tripwire
    plt.axvline(x=anomaly_threshold, color='black', linestyle='dashed', linewidth=2,
                label=f'Threshold ({anomaly_threshold})')

    # Format for Lab Report
    plt.xlabel('Mean Squared Error (MSE)', fontsize=12)
    plt.ylabel('Frequency (Number of Occurrences)', fontsize=12)
    plt.title('Edge AI Anomaly Detection: MSE Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    # Save the file to your directory
    plt.savefig('evaluation_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save the file to your directory
    plt.savefig('evaluation_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- AUTO-LOGGER INTEGRATION ---
    model_path = "../../Trash/frozen.pth"
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    log_benchmark(
        model_id="Model 1",
        architecture="1D Autoencoder",
        precision="float32",
        metric_name="Avg Anomaly MSE",
        metric_value=avg_anomaly,
        latency_ms=0.05,  # Placeholder baseline latency
        batch_size=1,
        model_size_mb=model_size_mb
    )


if __name__ == "__main__":
    run_evaluation_summary()