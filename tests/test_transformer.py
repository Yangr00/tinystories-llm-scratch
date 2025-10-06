import pytest
import torch
from torch import Tensor
from adapters import run_transformer_block, run_transformer_lm

# Helper function for test_transformer_block
def get_block_weights(d_model: int, num_heads: int, d_ff: int, device) -> dict[str, Tensor]:
    """Helper function to create a dictionary of random weights for a single Transformer block."""
    return {
        'attn_norm_weights': torch.randn(d_model, device=device),
        'wq_weights': torch.randn(d_model, d_model, device=device),
        'wk_weights': torch.randn(d_model, d_model, device=device),
        'wv_weights': torch.randn(d_model, d_model, device=device),
        'wo_weights': torch.randn(d_model, d_model, device=device),
        'ffn_norm_weights': torch.randn(d_model, device=device),
        'w1_weights': torch.randn(d_ff, d_model, device=device),
        'w2_weights': torch.randn(d_model, d_ff, device=device),
        'w3_weights': torch.randn(d_ff, d_model, device=device),
    }

# Helper function for test_transformer_lm
def get_flat_block_weights(layer_idx: int, d_model: int, num_heads: int, d_ff: int, device) -> dict[str, Tensor]:
    """Helper function to create a flattened dictionary of weights for a single Transformer block."""
    return {
        f'block_{layer_idx}_attn_norm_weights': torch.randn(d_model, device=device),
        f'block_{layer_idx}_wq_weights': torch.randn(d_model, d_model, device=device),
        f'block_{layer_idx}_wk_weights': torch.randn(d_model, d_model, device=device),
        f'block_{layer_idx}_wv_weights': torch.randn(d_model, d_model, device=device),
        f'block_{layer_idx}_wo_weights': torch.randn(d_model, d_model, device=device),
        f'block_{layer_idx}_ffn_norm_weights': torch.randn(d_model, device=device),
        f'block_{layer_idx}_w1_weights': torch.randn(d_ff, d_model, device=device),
        f'block_{layer_idx}_w2_weights': torch.randn(d_model, d_ff, device=device),
        f'block_{layer_idx}_w3_weights': torch.randn(d_ff, d_model, device=device),
    }

def test_transformer_block():
    """Smoke test for the TransformerBlock."""
    d_model, num_heads, batch_size, seq_len = 128, 4, 2, 64
    d_ff = int(8 / 3 * d_model)
    d_ff = (d_ff + 63) // 64 * 64
    device = torch.device("cpu")
    
    in_features = torch.randn(batch_size, seq_len, d_model, device=device)
    block_weights = get_block_weights(d_model, num_heads, d_ff, device)
    
    output = run_transformer_block(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=seq_len,
        theta=10000.0,
        weights=block_weights,
        in_features=in_features
    )
    
    assert output.shape == in_features.shape
    assert not torch.isnan(output).any()

def test_transformer_lm():
    """Smoke test for the full TransformerLM."""
    vocab_size, context_length, d_model, num_layers, num_heads = 512, 128, 256, 4, 4
    d_ff = int(8 / 3 * d_model)
    d_ff = (d_ff + 63) // 64 * 64
    batch_size, seq_len = 2, 64
    device = torch.device("cpu")
    
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Generate a single flat weights dictionary for the entire model
    weights = {
        'embedding_weights': torch.randn(vocab_size, d_model, device=device),
        'final_norm_weights': torch.randn(d_model, device=device),
        'output_layer_weights': torch.randn(vocab_size, d_model, device=device),
    }
    # Update the dictionary with weights from all blocks
    for i in range(num_layers):
        weights.update(get_flat_block_weights(i, d_model, num_heads, d_ff, device))

    logits = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=10000.0,
        weights=weights,
        in_indices=token_ids,
    )

    expected_shape = (batch_size, seq_len, vocab_size)
    assert logits.shape == expected_shape
    assert not torch.isnan(logits).any()