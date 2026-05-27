import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn

# 1. Import your blueprint and data
from models.mlp_encoder import MLPAutoencoder
from data.synthetic_power import SyntheticPowerDataset


def run_anomaly_detector():
    print("Booting Edge AI Anomaly Detector...")

    # 2. Load the Blueprint and the Learned Weights
    # Change to frozen.pth to change data that is used in evaluation
    model = MLPAutoencoder()
    model.load_state_dict(torch.load("trained_autoencoder.pth", weights_only=True))

    # CRITICAL: Tell PyTorch this is for deployment, not training!
    model.eval()

    # 3. Setup the Math
    criterion = nn.MSELoss()
    anomaly_threshold = 0.1  # Our trigger wire

    # 4. Generate "Live" Data (Matching Training Resolution!)
    dataset = SyntheticPowerDataset(num_samples=1000, sequence_length=64)

    # Grab a perfectly normal sequence
    normal_sensor_reading = dataset[0]

    # Clone it and inject the Day 1 Motor Stall Outlier (* 5.0)
    anomalous_sensor_reading = normal_sensor_reading.clone()
    anomalous_sensor_reading[20:25] = anomalous_sensor_reading[20:25] + 5.0

    # Manually add the "Batch" dimension so shape is [1, 64]
    normal_sensor_reading = normal_sensor_reading.unsqueeze(0)
    anomalous_sensor_reading = anomalous_sensor_reading.unsqueeze(0)

    # 5. Run Inference
    with torch.no_grad():
        print("\n--- Testing Normal Operations ---")
        normal_recon = model(normal_sensor_reading)
        normal_loss = criterion(normal_recon, normal_sensor_reading).item()

        print(f"Real-time MSE: {normal_loss:.4f}")
        if normal_loss > anomaly_threshold:
            print("STATUS: [ALERT] Anomaly Detected!")
        else:
            print("STATUS: [OK] System Healthy.")

        print("\n--- Testing Motor Stall ---")
        anomaly_recon = model(anomalous_sensor_reading)
        anomaly_loss = criterion(anomaly_recon, anomalous_sensor_reading).item()

        print(f"Real-time MSE: {anomaly_loss:.4f}")
        if anomaly_loss > anomaly_threshold:
            print("STATUS: [ALERT] Anomaly Detected!")
        else:
            print("STATUS: [OK] System Healthy.")

if __name__ == "__main__":
    run_anomaly_detector()