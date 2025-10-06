# tests/test_embedding.py
import torch
from adapters import run_embedding

def test_embedding():
    """
    Tests the Embedding module adapter.
    """
    # Define the dimensions for the test
    vocab_size = 100
    d_model = 64
    
    # Create random token IDs to look up
    token_ids = torch.randint(0, vocab_size, (4, 16)) # Batch of 4 sequences of 16 tokens

    # Create a random embedding weight matrix
    weights = torch.randn(vocab_size, d_model)

    # Get the output from your implementation via the adapter
    your_output = run_embedding(vocab_size, d_model, weights, token_ids)

    # Calculate the expected output using simple indexing
    expected_output = weights[token_ids]

    # Assert that your module's output is identical to the expected output
    assert torch.equal(your_output, expected_output), "Embedding module output is incorrect"