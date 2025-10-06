import pytest
import torch
from adapters import run_multihead_self_attention
from modules import RotaryPositionalEmbedding
from functional import scaled_dot_product_attention

@pytest.mark.parametrize(
    "batch_size, seq_len, d_model, n_heads",
    [
        (1, 64, 128, 4),
        (4, 256, 512, 8),
    ],
)
def test_multihead_self_attention(batch_size, seq_len, d_model, n_heads):
    """
    Tests the MultiHeadSelfAttention implementation by comparing its output
    with a manual, step-by-step calculation.
    """
    # --- 1. Setup ---
    d_k = d_v = d_model // n_heads
    d_in = d_out = d_model
    
    # Create random input tensor. It will be on the default device (CPU or GPU).
    in_features = torch.randn(batch_size, seq_len, d_in)
    # Get the device to ensure all other tensors are created on the same one.
    device = in_features.device
    
    # Create single-head weight tensors on the SAME device
    q_proj_weight = torch.randn(d_k, d_in, device=device)
    k_proj_weight = torch.randn(d_k, d_in, device=device)
    v_proj_weight = torch.randn(d_v, d_in, device=device)
    o_proj_weight = torch.randn(d_out, d_v, device=device)
    
    # --- 2. Manual Calculation (Ground Truth) ---
    # Manually construct the full weight matrices
    full_wq = q_proj_weight.repeat(n_heads, 1)
    full_wk = k_proj_weight.repeat(n_heads, 1)
    full_wv = v_proj_weight.repeat(n_heads, 1)
    full_wo = o_proj_weight.repeat(1, n_heads)

    # Project inputs
    q_proj = torch.matmul(in_features, full_wq.T)
    k_proj = torch.matmul(in_features, full_wk.T)
    v_proj = torch.matmul(in_features, full_wv.T)
    
    # Reshape and transpose
    q = q_proj.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
    k = k_proj.view(batch_size, seq_len, n_heads, d_k).transpose(1, 2)
    v = v_proj.view(batch_size, seq_len, n_heads, d_v).transpose(1, 2)
    
    # Apply RoPE - Ensure RoPE instance and token_positions are on the SAME device
    token_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    rope = RotaryPositionalEmbedding(theta=10000.0, d_k=d_k, max_seq_len=4096, device=device)
    q_rope = rope(q, token_positions)
    k_rope = rope(k, token_positions)
    
    # Create causal mask on the SAME device
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    
    # Scaled dot-product attention
    attn_out = scaled_dot_product_attention(q_rope, k_rope, v, mask=mask)
    
    # Concatenate and project
    attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
    expected_output = torch.matmul(attn_out, full_wo.T)
    
    # --- 3. Get Adapter Output ---
    actual_output = run_multihead_self_attention(
        d_model=d_model,
        num_heads=n_heads,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_features,
    )

    # --- 4. Compare ---
    assert actual_output.shape == expected_output.shape
    assert torch.allclose(actual_output, expected_output, atol=1e-5)