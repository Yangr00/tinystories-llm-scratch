# modules.py
import torch
from torch import nn
from functional import scaled_dot_product_attention

class Linear(nn.Module):
    """
    A Linear layer implementation without bias.
    """
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Construct a linear transformation module.

        Args:
            in_features (int): Final dimension of the input.
            out_features (int): Final dimension of the output.
            device (torch.device | None): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        
        # We store the weight matrix as (d_out, d_in)
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights using truncated normal distribution."""
        # A common initialization scheme for transformers
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input.

        Args:
            x (torch.Tensor): Input tensor of shape (..., in_features).

        Returns:
            torch.Tensor: Output tensor of shape (..., out_features).
        """
        # The operation is x @ W.T
        return x @ self.weight.t()

class Embedding(nn.Module):
    """
    An Embedding layer implementation.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        Construct an embedding module.

        Args:
            num_embeddings (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embedding vectors (d_model).
            device (torch.device | None): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # The embedding matrix has shape (vocab_size, d_model)
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights using truncated normal distribution."""
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.

        Args:
            token_ids (torch.Tensor): Tensor containing indices into the embedding matrix.

        Returns:
            torch.Tensor: The retrieved embeddings.
        """
        # Simple indexing into the embedding matrix
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module.

        Args:
            d_model (int): Hidden dimension of the model.
            eps (float): Epsilon value for numerical stability. Defaults to 1e-5.
            device (torch.device | None): Device to store the parameters on. Defaults to None.
            dtype (torch.dtype | None): Data type of the parameters. Defaults to None.
        """
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.d_model = d_model
        self.eps = eps
        
        # Learnable gain parameter, initialized to ones
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor and return a tensor of the same shape.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: Normalized and scaled tensor of the same shape.
        """
        # Store original dtype and upcast to float32 for precision
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Calculate the Root Mean Square of the input
        # We perform the mean calculation over the last dimension
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # Normalize the input and apply the learnable gain
        normalized_x = x / rms
        result = normalized_x * self.weight
        
        # Return the result in the original dtype
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    """
    The SwiGLU feed-forward network.
    """
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # The three linear layers for the SwiGLU formula
        self.w1 = Linear(in_features=d_model, out_features=d_ff, **factory_kwargs)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, **factory_kwargs)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN_SwiGLU(x) = W2(SiLU(W1(x)) * W3(x))
        xW1 = self.w1(x)
        silu_xW1 = xW1 * torch.sigmoid(xW1)
        xW3 = self.w3(x)
        gated_result = silu_xW1 * xW3
        return self.w2(gated_result)


class RotaryPositionalEmbedding(nn.Module):
    """
    Implements Rotary Positional Embedding (RoPE).
    """
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE.")

        # Precompute theta_i values on the correct device
        theta_values = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        
        # Precompute position values on the correct device
        m = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # Create outer product for m * theta_i
        m_theta = torch.outer(m, theta_values)

        # Precompute cos and sin and register as non-parameter buffers
        self.register_buffer("cos_cached", torch.cos(m_theta))
        self.register_buffer("sin_cached", torch.sin(m_theta))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        # Get cos and sin values for the given positions.
        # cos_cached shape: (max_seq_len, d_k/2)
        # token_positions shape: (batch, seq_len)
        # After indexing, cos/sin will have shape (batch, seq_len, d_k/2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # If x has a heads dimension, add a dimension to cos/sin for broadcasting.
        if x.dim() == 4:
            # New shape: (batch, 1, seq_len, d_k/2)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        # Reshape x to separate the pairs of values that will be rotated.
        x_pairs = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x_pairs.unbind(dim=-1)

        # Apply the rotation formula. cos and sin broadcast correctly.
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # Combine the rotated pairs back into the original shape.
        rotated_x = torch.stack((rotated_x1, rotated_x2), dim=-1).reshape_as(x)
        
        return rotated_x.to(in_dtype)


class MultiHeadSelfAttention(nn.Module):
    """
    Implements causal multi-head self-attention with RoPE.
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.wq = Linear(d_model, d_model, **factory_kwargs)
        self.wk = Linear(d_model, d_model, **factory_kwargs)
        self.wv = Linear(d_model, d_model, **factory_kwargs)
        self.wo = Linear(d_model, d_model, **factory_kwargs)
        
        self.rope = RotaryPositionalEmbedding(
            theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)
        
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        attention_output = scaled_dot_product_attention(q, k, v, mask=mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.wo(attention_output)



# In modules.py, append these two new classes

class TransformerBlock(nn.Module):
    """
    A single block of the Transformer, implementing the pre-norm architecture.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.norm1 = RMSNorm(d_model, **factory_kwargs)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, **factory_kwargs)
        self.norm2 = RMSNorm(d_model, **factory_kwargs)
        self.ffn = SwiGLU(d_model, d_ff, **factory_kwargs)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), token_positions)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    """
    The full Transformer Language Model.
    """
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # --- MODIFICATION: Store key config values on the model itself ---
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        # --- END MODIFICATION ---

        self.token_embedding = Embedding(vocab_size, d_model, **factory_kwargs)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, **factory_kwargs) 
            for _ in range(num_layers)
        ])
        
        self.final_norm = RMSNorm(d_model, **factory_kwargs)
        self.output_layer = Linear(d_model, vocab_size, **factory_kwargs)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_indices.shape
        device = in_indices.device
        
        x = self.token_embedding(in_indices)
        token_positions = torch.arange(seq_len, device=device).expand(batch_size, -1)
        
        for block in self.transformer_blocks:
            x = block(x, token_positions)
        
        x = self.final_norm(x)
        logits = self.output_layer(x)
        return logits