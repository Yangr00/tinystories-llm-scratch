import pytest
import torch
from adapters import run_rmsnorm

@pytest.mark.parametrize(
    "shape",
    [
        (16, 64, 256),    # A typical batch
        (32, 128, 768),   # A larger, common transformer size
        (8, 1, 1024),     # A batch of single tokens
        (1, 2048, 512),   # A very long sequence
        (1, 1, 128),      # A single vector
    ],
)
def test_rmsnorm_3d(shape):
    """
    Tests the RMSNorm implementation with various 3D input shapes by comparing
    its output with a manual calculation.
    """
    # The model dimension is the last element of the shape tuple
    d_model = shape[-1]
    eps = 1e-5

    # Create random tensors for the input features and the gain weights
    # These will be on the CPU by default, which is fine for a unit test
    in_features = torch.randn(shape)
    weights = torch.randn(d_model)

    # --- 1. Manual RMSNorm Calculation (This is our ground truth) ---
    # Upcast the input to float32 for the calculation, as per the instructions
    x_float = in_features.to(torch.float32)

    # Calculate the mean of the squares over the last dimension
    mean_of_squares = x_float.pow(2).mean(dim=-1, keepdim=True)
    
    # Calculate the reciprocal of the square root (rsqrt)
    rsqrt = torch.rsqrt(mean_of_squares + eps)
    
    # Normalize the input and apply the learnable gain (weights)
    expected_output = x_float * rsqrt * weights
    
    # Cast the result back to the original input data type
    expected_output = expected_output.to(in_features.dtype)


    # --- 2. Get the output from your `run_rmsnorm` adapter ---
    actual_output = run_rmsnorm(
        d_model=d_model,
        eps=eps,
        weights=weights,
        in_features=in_features,
    )

    # --- 3. Compare the results ---
    # First, ensure the output tensor has the exact same shape as the input
    assert actual_output.shape == expected_output.shape, "Output shape does not match expected shape."
    
    # Second, check that the values are numerically very close.
    # `torch.allclose` is used to account for tiny floating-point inaccuracies.
    assert torch.allclose(actual_output, expected_output, atol=1e-6), "Output values are not close enough to expected values."

def test_rmsnorm_2d():
    """Tests the RMSNorm implementation with a 2D input tensor."""
    shape = (10, 512) # A simple 2D shape (e.g., batch_size, d_model)
    d_model = shape[-1]
    eps = 1e-5

    in_features = torch.randn(shape)
    weights = torch.randn(d_model)

    # Manual calculation
    x_float = in_features.to(torch.float32)
    mean_of_squares = x_float.pow(2).mean(dim=-1, keepdim=True)
    rsqrt = torch.rsqrt(mean_of_squares + eps)
    expected_output = (x_float * rsqrt * weights).to(in_features.dtype)

    # Get output from your adapter
    actual_output = run_rmsnorm(
        d_model=d_model,
        eps=eps,
        weights=weights,
        in_features=in_features,
    )

    assert actual_output.shape == expected_output.shape
    assert torch.allclose(actual_output, expected_output, atol=1e-6)