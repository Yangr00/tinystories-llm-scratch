import pytest
import torch
import torch.nn.functional as F
from adapters import run_softmax

@pytest.mark.parametrize(
    "shape, dim",
    [
        ((16, 64, 256), -1),    # Standard 3D tensor, last dimension
        ((32, 128, 768), 1),    # 3D tensor, middle dimension
        ((8, 1024), 0),         # 2D tensor, first dimension
        ((1, 2048), -1),        # 2D tensor, last dimension
        ((4, 8, 128, 128), 2),  # 4D tensor (e.g., attention scores)
    ],
)
def test_softmax_matches_pytorch(shape, dim):
    """
    Tests that our custom softmax implementation produces the same output
    as the PyTorch built-in softmax function.
    """
    # --- 1. Setup ---
    # Create a random tensor with the specified shape
    in_features = torch.randn(shape)
    
    # --- 2. Calculate the expected output using PyTorch's softmax ---
    expected_output = F.softmax(in_features, dim=dim)
    
    # --- 3. Get the output from your `run_softmax` adapter ---
    actual_output = run_softmax(
        in_features=in_features,
        dim=dim
    )
    
    # --- 4. Compare the results ---
    assert actual_output.shape == expected_output.shape, "Output shape does not match."
    assert torch.allclose(actual_output, expected_output, atol=1e-7), "Output values are not close enough."

def test_softmax_numerical_stability():
    """
    Tests the numerical stability of the softmax implementation by using
    large input values that could cause overflow without the max-subtraction trick.
    """
    # Create a tensor with large values
    in_features = torch.tensor([1000.0, 1001.0, 999.0])
    
    # The expected output from a stable softmax should be non-NaN.
    # After subtracting max (1001): [-1, 0, -2]
    # exps: [e^-1, e^0, e^-2] = [0.3678, 1.0, 0.1353]
    # sum: 1.5031
    # softmax: [0.2447, 0.6652, 0.0900]
    expected_output = torch.tensor([0.2447, 0.6652, 0.0900])
    
    # Get the output from your adapter
    actual_output = run_softmax(in_features, dim=0)

    # Check that the output contains no NaN or inf values
    assert not torch.isnan(actual_output).any(), "Output contains NaN values."
    assert not torch.isinf(actual_output).any(), "Output contains inf values."
    # Check that the values are close to the expected stable result
    assert torch.allclose(actual_output, expected_output, atol=1e-4)