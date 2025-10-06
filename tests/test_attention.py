import pytest
import torch
from adapters import run_scaled_dot_product_attention

def test_scaled_dot_product_attention():
    """
    Tests the scaled dot-product attention implementation with 3D tensors.
    """
    batch_size, seq_len_q, seq_len_kv, d_k, d_v = 4, 10, 10, 32, 64
    
    # --- 1. Setup ---
    q = torch.randn(batch_size, seq_len_q, d_k)
    k = torch.randn(batch_size, seq_len_kv, d_k)
    v = torch.randn(batch_size, seq_len_kv, d_v)
    
    # --- 2. Manual Calculation (Ground Truth) ---
    expected_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k**0.5)
    expected_probs = torch.nn.functional.softmax(expected_scores, dim=-1)
    expected_output = torch.matmul(expected_probs, v)
    
    # --- 3. Get Adapter Output ---
    # CORRECTED: Use the argument names Q, K, V from adapters.py
    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=None)
    
    # --- 4. Compare ---
    assert actual_output.shape == expected_output.shape
    assert torch.allclose(actual_output, expected_output, atol=1e-6)

def test_4d_scaled_dot_product_attention():
    """
    Tests the scaled dot-product attention implementation with 4D tensors,
    simulating a multi-head attention scenario.
    """
    batch_size, n_heads, seq_len_q, seq_len_kv, d_k, d_v = 4, 8, 10, 10, 16, 32
    
    # --- 1. Setup ---
    q = torch.randn(batch_size, n_heads, seq_len_q, d_k)
    k = torch.randn(batch_size, n_heads, seq_len_kv, d_k)
    v = torch.randn(batch_size, n_heads, seq_len_kv, d_v)
    
    # --- 2. Manual Calculation (Ground Truth) ---
    expected_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k**0.5)
    expected_probs = torch.nn.functional.softmax(expected_scores, dim=-1)
    expected_output = torch.matmul(expected_probs, v)
    
    # --- 3. Get Adapter Output ---
    # CORRECTED: Use the argument names Q, K, V from adapters.py
    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=None)
    
    # --- 4. Compare ---
    assert actual_output.shape == expected_output.shape
    assert torch.allclose(actual_output, expected_output, atol=1e-6)

def test_attention_with_mask():
    """
    Tests the attention implementation with a causal mask.
    """
    batch_size, seq_len, d_k, d_v = 2, 5, 16, 32
    
    # --- 1. Setup ---
    q = torch.randn(batch_size, seq_len, d_k)
    k = torch.randn(batch_size, seq_len, d_k)
    v = torch.randn(batch_size, seq_len, d_v)
    
    # Create a causal mask where True means "attend".
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
    
    # --- 2. Manual Calculation (Ground Truth) ---
    expected_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k**0.5)
    # Apply mask before softmax
    expected_scores.masked_fill_(~mask, -torch.inf)
    expected_probs = torch.nn.functional.softmax(expected_scores, dim=-1)
    expected_output = torch.matmul(expected_probs, v)

    # --- 3. Get Adapter Output ---
    # CORRECTED: Use the argument names Q, K, V from adapters.py
    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    
    # --- 4. Compare ---
    assert actual_output.shape == expected_output.shape
    assert torch.allclose(actual_output, expected_output, atol=1e-6)
    
    # Verify that masked positions have zero probability
    from functional import softmax 
    internal_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k**0.5)
    internal_scores = internal_scores.masked_fill(~mask, -torch.inf)
    actual_probs = softmax(internal_scores, dim=-1)
    
    # Check that probabilities are zero where the mask is False
    assert torch.all(actual_probs.masked_select(~mask.unsqueeze(0)) < 1e-9)