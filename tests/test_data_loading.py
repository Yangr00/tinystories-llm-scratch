# /scratch.global/ruan0073/llm_scratch/tests/test_data_loading.py
import pytest
import torch
import numpy as np
from adapters import run_get_batch

# We test on CPU and GPU if available to ensure device placement works correctly.
@pytest.mark.parametrize("device", ["cpu", "cuda:0" if torch.cuda.is_available() else "cpu"])
def test_get_batch(device):
    """
    Tests the get_batch function for correctness.
    """
    # 1. Create a simple, predictable dataset of token IDs
    data = np.arange(1000)
    batch_size = 8
    context_length = 128

    # 2. Call the function via the adapter
    x_batch, y_batch = run_get_batch(data, batch_size, context_length, device)

    # 3. Perform assertions to check for correctness

    # Check that the output tensor shapes are correct
    assert x_batch.shape == (batch_size, context_length), "Input batch shape is incorrect."
    assert y_batch.shape == (batch_size, context_length), "Target batch shape is incorrect."

    # Check that the tensors are on the requested device
    assert str(x_batch.device) == device, f"Input batch is not on the correct device. Expected {device} but got {x_batch.device}."
    assert str(y_batch.device) == device, f"Target batch is not on the correct device. Expected {device} but got {y_batch.device}."


    # This is the most important check:
    # Verify that the target (y_batch) is the input (x_batch) shifted by one token.
    # The last (context_length - 1) tokens of x should be identical to
    # the first (context_length - 1) tokens of y.
    torch.testing.assert_close(x_batch[:, 1:], y_batch[:, :-1])