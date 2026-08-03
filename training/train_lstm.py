
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Import your custom modules
from data.data_loader import get_energy_dataloaders
from models.forecasting_model import LSTMForecaster


def train():
    print("--- INITIALIZING LSTM TRAINING PIPELINE ---")

    # 1. Setup Hyperparameters
    lookback_window = 24
    batch_size = 32
    epochs = 20
    learning_rate = 0.001

    # 2. Load the Data
    train_loader, test_loader = get_energy_dataloaders(lookback_window, batch_size)

    # 3. Initialize Model, Loss Function, and Optimizer
    model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=1, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\n--- STARTING TRAINING LOOP ---")

    # 4. The Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            # Clear old gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(inputs)

            # Calculate error
            loss = criterion(predictions, targets)

            # Backpropagation (calculate gradients)
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()

        # Print average loss for this epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}] | MSE Loss: {avg_loss:.6f}")

    # 5. Save the trained model
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "model3_lstm.pth")
    torch.save(model.state_dict(), save_path)

    print(f"\n--- TANGIBLE ARTIFACT ---")
    print(f"Training Complete. Model weights saved to: {save_path}")


if __name__ == "__main__":
    train()