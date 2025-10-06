# tests/test_linear.py
import torch
from adapters import run_linear

def test_linear():
    """
    Tests the Linear module adapter.
    """
    # Define the dimensions for the test
    d_in = 16
    d_out = 32
    batch_size = 4

    # Create random test data
    weights = torch.randn(d_out, d_in)
    in_features = torch.randn(batch_size, d_in)

    # Get the output from your implementation via the adapter
    your_output = run_linear(d_in, d_out, weights, in_features)

    # Calculate the expected output using standard PyTorch operations
    expected_output = in_features @ weights.t()

    # Assert that your module's output is very close to the expected output
    assert torch.allclose(your_output, expected_output, atol=1e-6), "Linear module output is incorrect"