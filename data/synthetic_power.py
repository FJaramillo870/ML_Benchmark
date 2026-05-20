import torch
from torch.utils.data import Dataset


class SyntheticPowerDataset(Dataset):
    def __init__(self, num_samples=1000, sequence_length=64):
        # Generate the normal wave
        total_points = num_samples * sequence_length
        time_steps = torch.linspace(0, 100, steps=total_points)
        base_power = torch.sin(time_steps)
        noise = torch.randn(total_points) * 0.2

        # Reshape into batches of 64 data points
        self.data = (base_power + noise).view(num_samples, sequence_length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]