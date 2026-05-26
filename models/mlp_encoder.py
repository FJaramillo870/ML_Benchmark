import torch.nn as nn

class MLPAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # Input: [Batch, 1 Channel, 64 Length]
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=15, stride=2, padding=7),
            nn.ReLU(),
            # Output: [Batch, 8 Channels, 32 Length]

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=15, stride=2, padding=7),
            nn.ReLU()
            # Output: [Batch, 16 Channels, 16 Length] (Your Bottleneck!)
        )

        # The Decoder: Expands sequence length from 16 -> 32 -> 64
        self.decoder = nn.Sequential(
            # Input: [Batch, 16 Channels, 16 Length]
            nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=15, stride=2, padding=7, output_padding=1),
            nn.ReLU(),
            # Output: [Batch, 8 Channels, 32 Length]

            nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=15, stride=2, padding=7, output_padding=1)
        # Output: [Batch, 1 Channel, 64 Length]
        )

    def forward(self, x):
        # x arrives as [32, 64] (Batch, Length)

        # 1. Insert the "1 Channel" dimension in the MIDDLE (Index 1)
        x = x.unsqueeze(1)
        # x is now [32, 1, 64]

        # 2. Pass through the network
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # 3. Remove the "1 Channel" dimension from the MIDDLE (Index 1)
        return decoded.squeeze(1)
        # decoded is back to [32, 64]
