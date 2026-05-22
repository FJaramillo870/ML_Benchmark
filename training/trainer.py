import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 1. Import YOUR custom blueprint and data
from models.mlp_encoder import MLPAutoencoder
from data.synthetic_power import SyntheticPowerDataset


def train_model():
    print("Initializing Training Pipeline...")

    # 2. Setup Data
    dataset = SyntheticPowerDataset(num_samples=1000, sequence_length=64)
    data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    # 3. Initialize Model, Loss, and Optimizer
    model = MLPAutoencoder()

    # Mean Squared Error: The standard for regression and reconstruction.
    # It mathematically punishes the model for how far off its predicted wave is from the real wave.
    criterion = nn.MSELoss()

    # Adam Optimizer: The industry standard algorithm that physically updates the model's weights.
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. The Training Loop
    num_epochs = 20

    print("Starting Training...")
    for epoch in range(num_epochs):
        total_loss = 0

        # Notice: No (img, _) tuple! Your dataset only returns the raw sensor data.
        for sensor_batch in data_loader:
            # A. Forward pass: Try to reconstruct the wave
            recon = model(sensor_batch)

            # B. Calculate the error (Loss)
            loss = criterion(recon, sensor_batch)

            # C. Backward pass: Calculate gradients and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print the average loss for this epoch
        avg_loss = total_loss / len(data_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    print("Training Complete!")

    # Save it directly to whatever folder PyCharm is currently looking at
    torch.save(model.state_dict(), "trained_autoencoder.pth")
    print("Model saved!")

if __name__ == "__main__":
    train_model()