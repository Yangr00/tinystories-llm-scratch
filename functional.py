import torch
import math
from torch import Tensor
from jaxtyping import Float, Bool, Int
import numpy as np
import os
from typing import BinaryIO, IO, Any, Dict

def softmax(x: Float[Tensor, "... dim"], dim: int) -> Float[Tensor, "... dim"]:
    """
    Computes the softmax of a tensor along a specified dimension in a numerically stable way.

    Args:
        x (torch.Tensor): Input tensor. Can have any shape.
        dim (int): The dimension along which to compute the softmax.

    Returns:
        torch.Tensor: The output tensor with the same shape as the input, with the specified
                      dimension normalized into a probability distribution.
    """
    # For numerical stability, subtract the maximum value along the given dimension.
    # This ensures the largest input to exp() is 0, preventing overflow.
    max_values = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - max_values
    
    # Calculate the exponent of the stable values.
    exponentials = torch.exp(x_stable)
    
    # Calculate the sum of the exponents along the dimension.
    sum_exponentials = torch.sum(exponentials, dim=dim, keepdim=True)
    
    # Divide to get the final probabilities.
    return exponentials / sum_exponentials

def scaled_dot_product_attention(
    q: Float[Tensor, "... seq_len_q d_k"],
    k: Float[Tensor, "... seq_len_kv d_k"],
    v: Float[Tensor, "... seq_len_kv d_v"],
    mask: Bool[Tensor, "seq_len_q seq_len_kv"] | None = None,
) -> Float[Tensor, "... seq_len_q d_v"]:
    """
    Computes scaled dot-product attention.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        v (torch.Tensor): Value tensor.
        mask (torch.Tensor | None): Optional boolean mask. A value of True indicates
                                    the position should be attended to. Defaults to None.

    Returns:
        torch.Tensor: The output of the attention mechanism.
    """
    # 1. Calculate the raw attention scores (Q * K^T)
    # The transpose swaps the last two dimensions of the key tensor.
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1))
    
    # 2. Scale the scores by the square root of d_k
    scaled_scores = scores / math.sqrt(d_k)
    
    # 3. Apply the mask if provided
    if mask is not None:
        # The mask is broadcasted to the shape of scaled_scores.
        # We add a very large negative number (-inf) where the mask is False.
        scaled_scores = scaled_scores.masked_fill(mask == False, -torch.inf)
        
    # 4. Apply the softmax function to get attention probabilities
    attention_probs = softmax(scaled_scores, dim=-1)
    
    # 5. Multiply the attention probabilities by the value tensor (V)
    output = torch.matmul(attention_probs, v)
    
    return output

def get_batch(
    x: Int[np.ndarray, " n"],
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[Int[Tensor, "batch_size context_length"], Int[Tensor, "batch_size context_length"]]:
    """
    Generates a random batch of input-target pairs from the data.

    Args:
        x (np.ndarray): The full tokenized data, a 1D numpy array of token IDs.
        batch_size (int): The number of sequences in a batch.
        context_length (int): The length of each sequence.
        device (str or torch.device): The device to place the output tensors on (e.g., 'cpu', 'cuda').

    Returns:
        A tuple of two tensors:
        - The input sequences (x_batch)
        - The target sequences (y_batch)
    """
    # 1. Generate random starting indices for each sequence in the batch
    # The highest possible starting index is len(x) - context_length - 1
    # to ensure there is a valid target for the last token.
    ix = torch.randint(len(x) - context_length, (batch_size,))

    # 2. Create the input sequences by slicing the data
    # For each starting index in `ix`, we take a slice of `context_length`
    x_batch = torch.stack([torch.from_numpy(x[i : i + context_length].astype(np.int64)) for i in ix])

    # 3. Create the target sequences, which are shifted by one token
    # For each starting index in `ix`, the target is the slice starting from `i+1`
    y_batch = torch.stack([torch.from_numpy(x[i + 1 : i + 1 + context_length].astype(np.int64)) for i in ix])

    # 4. Move the tensors to the specified device
    return x_batch.to(device), y_batch.to(device)

# def save_checkpoint(
#     model: torch.nn.Module,
#     optimizer: torch.optim.Optimizer,
#     iteration: int,
#     out: str | os.PathLike | BinaryIO | IO[bytes],
# ):
#     """
#     Dumps the model and optimizer state to a file-like object.

#     Args:
#         model (torch.nn.Module): The model to save.
#         optimizer (torch.optim.Optimizer): The optimizer to save.
#         iteration (int): The current iteration number.
#         out (str | os.PathLike | typing.BinaryIO | typing.IO[bytes]): The output path or file-like object.
#     """
#     checkpoint = {
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'iteration': iteration,
#     }
#     torch.save(checkpoint, out)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    config: Dict[str, Any], # <-- Add config argument
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Dumps the model, optimizer, and config to a file-like object.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'config': config, # <-- Save the config
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, Dict[str, Any]]:
    """
    Loads a checkpoint from a source and restores the model and optimizer states.

    Args:
        src (str | os.PathLike | typing.BinaryIO | typing.IO[bytes]): The source path or file-like object.
        model (torch.nn.Module): The model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.

    Returns:
        int: The iteration number saved in the checkpoint.
    """
    checkpoint = torch.load(src, map_location='cpu') # Load to CPU first
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Return the iteration and the config dictionary
    return checkpoint['iteration'], checkpoint.get('config', {})