import pytest
import torch
from adapters import run_rope

@pytest.mark.parametrize(
    "batch_size, seq_len, d_k",
    [
        (1, 128, 64),
        (4, 512, 128),
        (8, 64, 32),
    ],
)
def test_rope(batch_size, seq_len, d_k):
    """
    Tests the RoPE implementation by comparing its output with a manual calculation.
    """
    # --- 1. Setup ---
    theta = 10000.0
    max_seq_len = 2048 # A reasonable max length for testing
    
    in_features = torch.randn(batch_size, seq_len, d_k)
    # Generate token positions, e.g., [0, 1, 2, ..., seq_len-1] for each batch item
    token_positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # --- 2. Manual RoPE Calculation (Ground Truth) ---
    x = in_features.float()
    
    # Calculate frequencies
    freqs = theta ** (-torch.arange(0, d_k, 2).float() / d_k)
    
    # Calculate m * theta_i for all positions and frequencies
    m_theta = torch.outer(token_positions.flatten(), freqs)
    m_theta = m_theta.reshape(batch_size, seq_len, d_k // 2)
    
    # Calculate cos and sin values
    cos_vals = torch.cos(m_theta)
    sin_vals = torch.sin(m_theta)
    
    # Reshape x to separate pairs
    x_pairs = x.reshape(batch_size, seq_len, d_k // 2, 2)
    x1, x2 = x_pairs.unbind(-1)
    
    # Apply rotation to each pair
    rotated_x1 = x1 * cos_vals - x2 * sin_vals
    rotated_x2 = x1 * sin_vals + x2 * cos_vals
    
    # Combine pairs back
    expected_output = torch.stack((rotated_x1, rotated_x2), dim=-1).reshape_as(x)
    expected_output = expected_output.to(in_features.dtype)

    # --- 3. Get the output from your `run_rope` adapter ---
    # This is the corrected function call
    actual_output = run_rope(
        d_k=d_k,
        theta=theta,
        max_seq_len=max_seq_len,
        in_query_or_key=in_features,
        token_positions=token_positions,
    )
    
    # --- 4. Compare the results ---
    assert actual_output.shape == expected_output.shape, "Output shape mismatch."
    assert torch.allclose(actual_output, expected_output, atol=1e-5), "Output values do not match."