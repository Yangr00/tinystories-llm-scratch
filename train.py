# /scratch.global/ruan0073/llm_scratch/train.py
import os
import argparse
import time
import math  # <-- This was the missing import
import torch
from torch import nn, optim
import numpy as np
from modules import TransformerLM
from functional import get_batch, save_checkpoint, load_checkpoint
from tokenizer import Tokenizer
import config as C

# --- Learning Rate Scheduler Logic ---
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < C.warmup_iters:
        return C.learning_rate * it / C.warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > C.max_iters:
        return C.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - C.warmup_iters) / (C.max_iters - C.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return C.min_lr + coeff * (C.learning_rate - C.min_lr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Transformer Language Model')
    parser.add_argument('--train_data_path', type=str, required=True, help='Path to tokenized training data (.npy)')
    parser.add_argument('--val_data_path', type=str, required=True, help='Path to tokenized validation data (.npy)')
    parser.add_argument('--vocab_path', type=str, default='vocab.json', help='Path to the vocabulary file.')
    parser.add_argument('--merges_path', type=str, default='merges.txt', help='Path to the merges file.')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to a checkpoint to resume training from')
    args = parser.parse_args()

    print(f"Using device: {C.device}")
    os.makedirs(C.checkpoint_dir, exist_ok=True)
    
    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.val_data_path, dtype=np.uint16, mode='r')
    
    tokenizer = Tokenizer.from_files(vocab_filepath=args.vocab_path, merges_filepath=args.merges_path, special_tokens=["<|endoftext|>"])
    actual_vocab_size = len(tokenizer.vocab)
    print(f"Actual vocabulary size: {actual_vocab_size}")

    model_args = {
        'vocab_size': actual_vocab_size,
        'context_length': C.context_length,
        'd_model': C.d_model,
        'num_layers': C.num_layers,
        'num_heads': C.num_heads,
        'd_ff': C.d_ff,
        'rope_theta': C.rope_theta,
    }
    model = TransformerLM(**model_args).to(C.device)
    optimizer = optim.AdamW(model.parameters(), lr=C.learning_rate, weight_decay=C.weight_decay, betas=(C.beta1, C.beta2))
    loss_fn = nn.CrossEntropyLoss()

    start_iter = 0
    if args.load_checkpoint:
        start_iter, _ = load_checkpoint(args.load_checkpoint, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    print("Starting training...")
    start_time = time.time()
    
    for i in range(start_iter, C.max_iters):
        # Set learning rate for this iteration
        lr = get_lr(i)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        model.train()
        x, y = get_batch(train_data, C.batch_size, C.context_length, C.device)
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % C.log_interval == 0:
            elapsed_time = time.time() - start_time
            print(f"Iteration {i:5d}/{C.max_iters} | LR: {lr:.6f} | Loss: {loss.item():.4f} | Time: {elapsed_time:.2f}s")

        if i > 0 and i % C.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_loss_accum = 0
                for _ in range(10): 
                    x_val, y_val = get_batch(val_data, C.batch_size, C.context_length, C.device)
                    val_logits = model(x_val)
                    val_loss = loss_fn(val_logits.view(-1, val_logits.size(-1)), y_val.view(-1))
                    val_loss_accum += val_loss.item()
                avg_val_loss = val_loss_accum / 10
                print(f"--- Validation Loss at Iter {i}: {avg_val_loss:.4f} ---")
            
            checkpoint_path = os.path.join(C.checkpoint_dir, f'ckpt_iter_{i}.pth')
            print(f"Saving checkpoint to {checkpoint_path}")
            save_checkpoint(model, optimizer, i, model_args, checkpoint_path)
    
    print("--- Training Finished ---")
    final_checkpoint_path = os.path.join(C.checkpoint_dir, f'ckpt_iter_{C.max_iters}.pth')
    print(f"Saving final checkpoint to {final_checkpoint_path}")
    save_checkpoint(model, optimizer, C.max_iters, model_args, final_checkpoint_path)