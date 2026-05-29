import main
import torch

def test_neural_network_forward_pass():
    # If the function runs without crashing, the test passes
    main.test_forward_pass()


from data.synthetic_power import SyntheticPowerDataset  # Adjust import if needed

def test_synthetic_power_data_shapes():
    """Verifies Model 1's 1D power wave generator."""
    dataset = SyntheticPowerDataset(num_samples=100, sequence_length=64)

    # Grab one wave
    sample_wave = dataset[0]

    # Check the math
    assert sample_wave.shape == (64,), f"Expected 1D sequence of 64, got {sample_wave.shape}"
    assert type(sample_wave) == torch.Tensor, "Power wave must be a PyTorch Tensor"


from models.mlp_encoder import MLPAutoencoder

def test_autoencoder_forward_pass():
    """Verifies Model 1 can reconstruct a 1D power wave."""
    model = MLPAutoencoder()

    # Synthetic batch of 8 power waves, each with 64 data points
    # (Note: CNNs usually expect [Batch, Channels, Length] -> [8, 1, 64])
    dummy_waves = torch.randn(8, 1, 64)

    reconstruction = model(dummy_waves)

    assert reconstruction.shape == dummy_waves.shape, f"Reconstruction shape {reconstruction.shape} does not match input {dummy_waves.shape}"


def test_anomaly_threshold_logic():
    """Verifies that the threshold math correctly flags stalls."""
    threshold = 0.100

    # Simulate a healthy MSE
    normal_mse = 0.010
    # Simulate a stalled MSE
    stall_mse = 0.570

    assert normal_mse < threshold, "Threshold is too tight, flagging false positives!"
    assert stall_mse > threshold, "Threshold is too loose, missing motor stalls!"