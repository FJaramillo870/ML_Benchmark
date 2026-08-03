import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class SlidingWindowDataset(Dataset):
    def __init__(self, data_array, lookback_window, forecast_horizon=1):
        """
        Args:
            data_array (numpy.ndarray): The normalized time-series data.
            lookback_window (int): How many past time steps to look at.
            forecast_horizon (int): How many future time steps to predict.
        """
        # Convert numpy array to PyTorch tensor
        self.data = torch.FloatTensor(data_array)
        self.lookback = lookback_window
        self.horizon = forecast_horizon

    def __len__(self):
        # We can only slide the window until we hit the end of the data
        return len(self.data) - self.lookback - self.horizon + 1

    def __getitem__(self, idx):
        # Input (X): The historical data window
        x = self.data[idx: idx + self.lookback]

        # Target (Y): The future value we want to predict
        # We are predicting the first column (Energy Usage)
        y = self.data[idx + self.lookback: idx + self.lookback + self.horizon, 0]

        return x, y


def get_energy_dataloaders(lookback_window=24, batch_size=32):
    print("Downloading dataset...")
    # Fetching a reliable, open-source daily temperature/energy dataset
    # (Using a classic time-series dataset from standard raw GitHub repositories for reliability)
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    df = pd.read_csv(url, usecols=[1])  # Grab just the metric column

    # 1. Normalize the data (Neural networks struggle with large raw numbers)
    raw_data = df.values.astype(np.float32)
    data_min = np.min(raw_data)
    data_max = np.max(raw_data)
    normalized_data = (raw_data - data_min) / (data_max - data_min)

    # 2. Split into Train (80%) and Test (20%)
    split_idx = int(len(normalized_data) * 0.8)
    train_data = normalized_data[:split_idx]
    test_data = normalized_data[split_idx:]

    # 3. Create the PyTorch Datasets
    train_dataset = SlidingWindowDataset(train_data, lookback_window)
    test_dataset = SlidingWindowDataset(test_data, lookback_window)

    # 4. Create the PyTorch DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    # Test the pipeline
    print("Building DataLoaders...")
    train_loader, test_loader = get_energy_dataloaders(lookback_window=24, batch_size=16)

    # Grab one batch to verify
    for inputs, targets in train_loader:
        print("\n--- TANGIBLE ARTIFACT ---")
        print(f"Input X shape:  {inputs.shape}  -> [Batch Size, Lookback Window, Features]")
        print(f"Target Y shape: {targets.shape}  -> [Batch Size, Forecast Horizon]")

        print("\nFirst sequence targets (Y):")
        print(targets[0:5])  # Print first 5 targets in the batch
        break