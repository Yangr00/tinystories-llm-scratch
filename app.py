import torch
import gradio as gr
from modules import TransformerLM
from tokenizer import Tokenizer
from functional import softmax

# --- Configuration ---
# Hardcode the model architecture. Ensure these match the model you are loading!
CONTEXT_LENGTH = 256
D_MODEL = 512
NUM_LAYERS = 4
NUM_HEADS = 16
D_FF = 1344
ROPE_THETA = 10000.0
CHECKPOINT_PATH = "ckpt_iter_49500.pth"
DEVICE = "cpu" # Hugging Face Spaces default to CPU

# --- Load Model and Tokenizer (once at startup) ---
print("Loading tokenizer...")
tokenizer = Tokenizer.from_files(
    vocab_filepath="vocab.json",
    merges_filepath="merges.txt",
    special_tokens=["<|endoftext|>"]
)
ACTUAL_VOCAB_SIZE = len(tokenizer.vocab)

print(f"Loading model from {CHECKPOINT_PATH}...")
model_args = {
    'vocab_size': ACTUAL_VOCAB_SIZE, 'context_length': CONTEXT_LENGTH, 'd_model': D_MODEL,
    'num_layers': NUM_LAYERS, 'num_heads': NUM_HEADS, 'd_ff': D_FF, 'rope_theta': ROPE_THETA,
}
model = TransformerLM(**model_args).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded successfully.")

# --- The Generation Function ---
@torch.no_grad()
def run_generation(prompt, max_new_tokens, temperature, top_p):
    # Ensure temperature is not zero
    if temperature <= 0:
        temperature = 1.0

    # Encode prompt and create tensor
    prompt_ids = tokenizer.encode(prompt)
    context_ids = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    for _ in range(max_new_tokens):
        current_ids = context_ids[:, -CONTEXT_LENGTH:]
        logits = model(current_ids)
        last_token_logits = logits[:, -1, :]
        
        scaled_logits = last_token_logits / temperature
        probs = softmax(scaled_logits, dim=-1)
        
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        indices_to_remove = cumulative_probs > top_p
        indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
        indices_to_remove[..., 0] = 0
        
        probs_to_remove = torch.zeros_like(probs)
        probs_to_remove.scatter_(dim=-1, index=sorted_indices, src=indices_to_remove.float())
        probs[probs_to_remove.bool()] = 0
        
        if torch.all(probs.sum(dim=-1) == 0):
            next_token_id = sorted_indices[..., :1]
        else:
            probs = probs / probs.sum(dim=-1, keepdim=True)
            next_token_id = torch.multinomial(probs, num_samples=1)
            
        context_ids = torch.cat([context_ids, next_token_id], dim=1)
        
        if next_token_id.item() == tokenizer.byte_to_token_id[b'<|endoftext|>']:
            break
            
    return tokenizer.decode(context_ids[0].tolist())

# --- Create the Gradio Interface ---
iface = gr.Interface(
    fn=run_generation,
    inputs=[
        gr.Textbox(lines=2, label="Prompt", placeholder="Once upon a time there was..."),
        gr.Slider(minimum=10, maximum=500, value=250, step=10, label="Max New Tokens"),
        gr.Slider(minimum=0.1, maximum=1.5, value=0.7, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.9, label="Top-p (Nucleus Sampling)"),
    ],
    outputs=gr.Textbox(label="Generated Story"),
    title="TinyStories LLM from Scratch",
    description="A ~17M parameter Transformer model built and trained from scratch. Enter a prompt to generate a story."
)

if __name__ == "__main__":
    iface.launch()