import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn


class CIFAR10Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        # --- THE FEATURE EXTRACTOR (The "Encoder") ---
        self.features = nn.Sequential(
            # Input: N, 3, 32, 32
            # 3 because its RGB, create 16 different "feature detectors", will use a 3x3 grid of pixels, moves 1 pixel at a time to the right until it hits the edge it moves down one pixel and does it again, creates a 1 pixel thick value of 0 so the full image can be scanned
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> N, 16, 16, 16

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> N, 32, 8, 8
        )

        # --- THE CLASSIFIER HEAD ---
        self.classifier = nn.Sequential(
            # We must flatten the 2D grids before feeding to Linear layers
            nn.Flatten(),

            # The math: 32 channels * 8 height * 8 width = 2048
            nn.Linear(in_features=32 * 8 * 8, out_features=128),
            nn.ReLU(),

            # Final output must be exactly 10 (for the 10 CIFAR classes)
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        # 1. Extract visual features
        x = self.features(x)

        print(f"Shape before Flatten: {x.shape}")

        # 2. Classify based on those features
        x = self.classifier(x)

        return x


# --- THE QUARANTINE ZONE: TESTING THE FORWARD PASS ---
if __name__ == "__main__":
    print("Booting Model 2 Architecture Test...")

    # 1. Initialize the blueprint
    model = CIFAR10Classifier()

    # 2. Create a dummy batch of 8 CIFAR-10 images
    # Shape matches your data loader: [Batch Size, Channels, Height, Width]
    dummy_batch = torch.randn(8, 3, 32, 32)

    print(f"Input Shape:  {dummy_batch.shape}")

    # 3. Run the Forward Pass
    predictions = model(dummy_batch)

    print(f"Output Shape: {predictions.shape}")

    # Verification logic
    if predictions.shape == (8, 10):
        print("STATUS: [SUCCESS] Forward pass completed. Architecture is sound.")
    else:
        print("STATUS: [FAIL] Dimension mismatch.")