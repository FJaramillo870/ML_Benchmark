import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim

# 1. Import your custom blueprint and data pipeline
from models.model2 import CIFAR10Classifier  # Ensure this matches your actual filename!
from data.vision_dataset import get_cifar10_loaders


def train_model():
    print("Initializing Vision Training Pipeline...")

    # 2. Setup Data using the modular function you already built
    # Increased batch_size to 32 so the epoch doesn't take forever
    train_loader, _ = get_cifar10_loaders(batch_size=32)

    # 3. Initialize Model, Loss, and Optimizer
    model = CIFAR10Classifier()

    # CRITICAL CHANGE: CrossEntropy is the industry standard for classification
    criterion = nn.CrossEntropyLoss()

    # Adam Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. The Training Loop
    num_epochs = 10  # 10 epochs is usually enough to see CIFAR-10 start learning

    print("Starting Training...")
    for epoch in range(num_epochs):
        total_loss = 0

        # CRITICAL CHANGE: Vision loaders return BOTH the image and the correct label!
        for images, labels in train_loader:
            # A. Forward pass: The model guesses the class (Outputs 10 probabilities)
            predictions = model(images)

            # B. Calculate the error (How far was the guess from the actual label?)
            loss = criterion(predictions, labels)

            # C. Backward pass: Calculate gradients and update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print the average loss for this epoch
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    print("Training Complete!")

    #Save it with a new name so you don't overwrite your anomaly detector!
    torch.save(model.state_dict(), "model2_cifar10.pth")
    print("Model 2 saved!")


if __name__ == "__main__":
    train_model()