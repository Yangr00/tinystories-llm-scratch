# /scratch.global/ruan0073/llm_scratch/config.py
import torch

# Configuration for a small ~17M parameter Transformer model

# --- Model Hyperparameters ---
# NOTE: vocab_size is the target size for tokenizer training.
# The actual vocab size will be determined by the trained tokenizer file.
vocab_size = 10000
context_length = 256       # Maximum sequence length.
d_model = 512              # Embedding dimension.
num_layers = 4             # Number of Transformer blocks.
num_heads = 16             # Number of attention heads. Must divide d_model.
d_ff = 1344                # Dimension of the feed-forward network. (Approx. 8/3 * d_model)
rope_theta = 10000.0       # RoPE theta parameter.

# --- Training Hyperparameters ---
batch_size = 32
learning_rate = 3e-4       # A common starting point for AdamW.
max_iters = 50000           # Total training iterations for a quick run.
weight_decay = 0.1         # Weight decay for the optimizer.
beta1 = 0.9                # AdamW beta1.
beta2 = 0.95               # AdamW beta2.
warmup_iters = 200         # How many steps to warm up for
min_lr = 3e-5              # The final learning rate after decay

# --- System & I/O ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_dir = 'checkpoints'
log_interval = 100         # Print training loss every N iterations.
val_interval = 500         # Run validation and save checkpoint every N iterations.