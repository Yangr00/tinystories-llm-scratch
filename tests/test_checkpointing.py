import pytest
import torch
from torch import nn, optim
import io
from adapters import run_save_checkpoint, run_load_checkpoint

# A simple dummy model for testing purposes
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

def test_checkpointing():
    """
    Tests that saving and loading a checkpoint restores the model and optimizer states.
    """
    # 1. Setup original model and optimizer
    original_model = SimpleModel()
    original_optimizer = optim.AdamW(original_model.parameters(), lr=0.001)
    original_iteration = 42

    # Advance the optimizer state by performing a few steps
    for _ in range(5):
        original_optimizer.zero_grad()
        loss = original_model(torch.randn(4, 10)).sum()
        loss.backward()
        original_optimizer.step()

    # 2. Save the checkpoint to an in-memory buffer
    buffer = io.BytesIO()
    run_save_checkpoint(original_model, original_optimizer, original_iteration, buffer)
    
    # Ensure the buffer is ready to be read from
    buffer.seek(0)

    # 3. Setup new model and optimizer to load into
    new_model = SimpleModel()
    new_optimizer = optim.AdamW(new_model.parameters(), lr=0.1) # Use different LR to check if it's restored

    # Verify they are different before loading
    assert not torch.equal(
        original_model.state_dict()['linear.weight'],
        new_model.state_dict()['linear.weight']
    )

    # 4. Load the checkpoint
    loaded_iteration = run_load_checkpoint(buffer, new_model, new_optimizer)

    # 5. Assert that states are now identical
    assert loaded_iteration == original_iteration, "Iteration number was not restored correctly."

    # Check model weights
    assert torch.equal(
        original_model.state_dict()['linear.weight'],
        new_model.state_dict()['linear.weight']
    ), "Model state_dict does not match after loading."

    # --- MODIFICATION START ---
    # Check optimizer state correctly
    # We cannot use a direct `==` comparison because the state_dict contains tensors.
    
    sd_orig = original_optimizer.state_dict()
    sd_new = new_optimizer.state_dict()

    # Compare param_groups (non-tensor part)
    assert sd_orig['param_groups'] == sd_new['param_groups'], "Optimizer param_groups do not match."

    # Compare state (the part with tensors)
    state_orig = sd_orig['state']
    state_new = sd_new['state']
    
    assert state_orig.keys() == state_new.keys(), "Optimizer state has different parameter keys."

    for param_id in state_orig.keys():
        param_state_orig = state_orig[param_id]
        param_state_new = state_new[param_id]
        
        for key in param_state_orig.keys():
            val_orig = param_state_orig[key]
            val_new = param_state_new[key]
            if isinstance(val_orig, torch.Tensor):
                assert torch.equal(val_orig, val_new), f"Tensor state for '{key}' in param {param_id} does not match."
            else:
                assert val_orig == val_new, f"Scalar state for '{key}' in param {param_id} does not match."
    # --- MODIFICATION END ---
