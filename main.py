import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
from torch.utils.data import DataLoader

# Import your custom classes from your folders
from data.synthetic_power import SyntheticPowerDataset
from models.mlp_encoder import MLPAutoencoder


def test_forward_pass():
    print("Initializing pipeline...")

    # 1. Load the fake sensor data
    dataset = SyntheticPowerDataset(num_samples=100, sequence_length=64)
    data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

    # 2. Load the model
    model = MLPAutoencoder()

    # 3. Grab one batch of data (32 sequences of 64 points)
    dataiter = iter(data_loader)
    sensor_batch = next(dataiter)

    # 4. Run the data through the model
    reconstructed_batch = model(sensor_batch)

    # 5. Verify the shape hasn't changed
    assert sensor_batch.shape == reconstructed_batch.shape

    print("Forward pass successful!")
    print(f"Input shape:  {sensor_batch.shape}")
    print(f"Output shape: {reconstructed_batch.shape}")


if __name__ == "__main__":
    test_forward_pass()