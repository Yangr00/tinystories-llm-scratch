# /scratch.global/ruan0073/llm_scratch/generate.py
import torch  
import argparse
from functional import softmax
from modules import TransformerLM
from tokenizer import Tokenizer
import config as C 

@torch.no_grad()
def generate(model, tokenizer, prompt_ids, max_new_tokens, temperature, top_p):
    """
    Generates a sequence of tokens given a prompt.
    """
    model.eval()
    
    # Get context_length directly from the model object
    context_length = model.context_length
    context_ids = prompt_ids
    
    for _ in range(max_new_tokens):
        # Crop context if it exceeds the model's context length
        current_ids = context_ids[:, -context_length:]
        
        # Get the model's predictions (logits) for the next token
        logits = model(current_ids)
        
        # We only care about the logits for the very last token in the sequence
        last_token_logits = logits[:, -1, :] # Shape: (batch_size, vocab_size)
        
        # Apply Temperature Scaling
        scaled_logits = last_token_logits / temperature
        
        # Get Probabilities via Softmax
        probs = softmax(scaled_logits, dim=-1)
        
        # Apply Top-p (Nucleus) Sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        indices_to_remove = cumulative_probs > top_p
        indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
        indices_to_remove[..., 0] = 0
        
        probs_to_remove = torch.zeros_like(probs)
        probs_to_remove.scatter_(dim=-1, index=sorted_indices, src=indices_to_remove.float())
        probs[probs_to_remove.bool()] = 0
        
        if torch.all(probs.sum(dim=-1) == 0):
            next_token_id = sorted_indices[..., :1] # Fallback to the most likely token
        else:
            probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token_id = torch.multinomial(probs, num_samples=1)

        # Append the sampled token to our context
        context_ids = torch.cat([context_ids, next_token_id], dim=1)
        
        # Stop if we generate the end-of-text token
        if next_token_id.item() == tokenizer.byte_to_token_id[b'<|endoftext|>']:
            break
            
    return context_ids

def main():
    parser = argparse.ArgumentParser(description='Generate text from a trained Transformer model.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint (.pth file).')
    parser.add_argument('--prompt', type=str, default='Once upon a time', help='The starting prompt for generation.')
    parser.add_argument('--max_new_tokens', type=int, default=100, help='Maximum number of new tokens to generate.')
    parser.add_argument('--temperature', type=float, default=0.7, help='Controls randomness. Higher is more random.')
    parser.add_argument('--top_p', type=float, default=0.9, help='Nucleus sampling threshold.')
    parser.add_argument('--vocab_path', type=str, default='vocab.json', help='Path to the vocabulary file.')
    parser.add_argument('--merges_path', type=str, default='merges.txt', help='Path to the merges file.')
    args = parser.parse_args()

    # --- 1. Load Checkpoint and Extract Config ---
    print(f"Loading model configuration from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=C.device)
    
    model_args = checkpoint.get('config')
    if model_args is None:
        print("Warning: No config found in checkpoint. Using default config.py values.")
        tokenizer = Tokenizer.from_files(vocab_filepath=args.vocab_path, merges_filepath=args.merges_path, special_tokens=["<|endoftext|>"])
        model_args = {
            'vocab_size': len(tokenizer.vocab), 'context_length': C.context_length, 'd_model': C.d_model,
            'num_layers': C.num_layers, 'num_heads': C.num_heads, 'd_ff': C.d_ff, 'rope_theta': C.rope_theta,
        }
    
    print("\n--- Model Configuration from Checkpoint ---")
    for key, value in model_args.items():
        print(f"{key:>15}: {value}")
    print("-------------------------------------------\n")

    # --- 2. Build Model and Load Weights ---
    model = TransformerLM(**model_args).to(C.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model weights loaded successfully.")

    # --- 3. Load Tokenizer ---
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens=["<|endoftext|>"]
    )
    
    # --- 4. Generate Text ---
    print("\n--- Generating Text ---")
    print(f"Prompt: {args.prompt}")
    prompt_ids = tokenizer.encode(args.prompt)
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=C.device).unsqueeze(0)
    
    generated_ids = generate(model, tokenizer, prompt_tensor, args.max_new_tokens, args.temperature, args.top_p)
    
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    print("\n--- Generated Output ---")
    print(generated_text)
    print("------------------------")

if __name__ == '__main__':
    main()
