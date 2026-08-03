import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super(LSTMForecaster, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 1. The LSTM Layer: Processes the time-series sequence
        # batch_first=True tells PyTorch our data is structured as [Batch, Sequence, Features]
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 2. The Output Layer: Maps the LSTM's final hidden state to a single prediction value
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [Batch Size, Lookback Window, Features] (e.g., [16, 24, 1])

        # Initialize the hidden state and cell state with zeros
        # This is strictly required for the first step of the sequence
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass the input through the LSTM
        # out shape: [Batch Size, Lookback Window, Hidden Size]
        # We also get the final hidden and cell states, but we often don't need them directly here
        out, _ = self.lstm(x, (h0, c0))

        # We only care about the LSTM's output at the VERY LAST time step to make our future prediction
        # out[:, -1, :] grabs all batches, the last sequence step, and all hidden features
        final_timestep_out = out[:, -1, :]

        # Pass that final state through the linear layer to get our 1 predicted value
        prediction = self.fc(final_timestep_out)

        return prediction


if __name__ == "__main__":
    print("Building Model 3 (LSTM)...")
    model = LSTMForecaster(input_size=1, hidden_size=32, num_layers=1, output_size=1)

    # Create a dummy input tensor matching the shape from yesterday's dataloader
    # [Batch Size = 16, Lookback Window = 24, Features = 1]
    dummy_input = torch.randn(16, 24, 1)

    print("\n--- TANGIBLE ARTIFACT ---")
    print(f"Dummy Input Shape:  {dummy_input.shape}")

    # Run the forward pass
    output = model(dummy_input)

    print(f"Prediction Shape:   {output.shape} -> [Batch Size, Forecast Horizon]")

    print("\nFirst 5 raw predictions (untrained):")
    print(output[0:5])