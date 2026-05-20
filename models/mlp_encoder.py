import torch.nn as nn

class MLPAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # The Encoder: Compresses 64 points down to 16
        self.encoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # The Decoder: Expands 16 points back to 64
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
            # Note: No Sigmoid here!
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded