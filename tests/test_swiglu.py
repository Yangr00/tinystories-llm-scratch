import pytest
import torch
from adapters import run_swiglu

@pytest.mark.parametrize(
    "d_model, batch_size, seq_len",
    [
        (256, 16, 64),
        (768, 8, 128),
        (512, 1, 1),
    ],
)
def test_swiglu(d_model, batch_size, seq_len):
    """
    Tests the SwiGLU implementation by comparing its output with a manual,
    direct calculation based on the formula.
    """
    # --- 1. Setup ---
    # Determine the hidden dimension d_ff using the required formula
    d_ff = int(8 / 3 * d_model)
    d_ff = (d_ff + 63) // 64 * 64
    
    # Create random tensors for the input and the three weight matrices
    in_features = torch.randn(batch_size, seq_len, d_model)
    # Note the new variable names to match the adapter signature
    w1_weight = torch.randn(d_ff, d_model)
    w3_weight = torch.randn(d_ff, d_model)
    w2_weight = torch.randn(d_model, d_ff)

    # --- 2. Manual SwiGLU Calculation (Ground Truth) ---
    # Formula: W2(SiLU(W1(x)) * W3(x))
    # Note: Our Linear layer does (x @ W.T).
    xW1 = torch.matmul(in_features, w1_weight.T)
    xW3 = torch.matmul(in_features, w3_weight.T)
    
    # Apply SiLU activation to xW1
    silu_xW1 = xW1 * torch.sigmoid(xW1)
    
    # Element-wise product
    gated_result = silu_xW1 * xW3
    
    # Final matrix multiplication
    expected_output = torch.matmul(gated_result, w2_weight.T)

    # --- 3. Get the output from your `run_swiglu` adapter ---
    actual_output = run_swiglu(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=w1_weight,
        w2_weight=w2_weight,
        w3_weight=w3_weight,
        in_features=in_features,
    )

    # --- 4. Compare the results ---
    assert actual_output.shape == expected_output.shape, "Output shape does not match expected shape."
    assert torch.allclose(actual_output, expected_output, atol=1e-5), "Output values are not close enough."