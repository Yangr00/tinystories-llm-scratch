# /scratch.global/ruan0073/llm_scratch/prepare_data.py
import os
import argparse
import numpy as np
from tokenizer import Tokenizer

def main():
    parser = argparse.ArgumentParser(description='Tokenize a text file and save as a .npy memory-mapped file.')
    parser.add_argument('--input_txt', type=str, required=True, help='Path to the input text file.')
    parser.add_argument('--output_npy', type=str, required=True, help='Path to save the output .npy file.')
    parser.add_argument('--vocab_path', type=str, default='vocab.json', help='Path to the vocabulary file.')
    parser.add_argument('--merges_path', type=str, default='merges.txt', help='Path to the merges file.')
    args = parser.parse_args()

    print("--- Starting Data Preparation ---")

    # 1. Load the trained tokenizer
    print(f"Loading tokenizer from {args.vocab_path} and {args.merges_path}...")
    if not os.path.exists(args.vocab_path) or not os.path.exists(args.merges_path):
        print("Error: vocab.json or merges.txt not found.")
        print("Please run train_tokenizer.py first to create these files.")
        return
        
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens=special_tokens
    )
    
    # 2. Read the input text file
    print(f"Reading input file: {args.input_txt}")
    with open(args.input_txt, 'r', encoding='utf-8') as f:
        text_data = f.read()
    
    # 3. Encode the entire text file
    print("Encoding text data... (this may take a moment)")
    token_ids = tokenizer.encode(text_data)
    print(f"Successfully encoded text into {len(token_ids):,} tokens.")
    
    # 4. Save to a memory-mapped numpy file
    print(f"Saving tokens to {args.output_npy}")
    # We use uint16 because it's memory-efficient for vocab sizes up to 65,535
    # For larger vocabs, you might need uint32
    mmap_array = np.memmap(args.output_npy, dtype=np.uint16, mode='w+', shape=(len(token_ids),))
    mmap_array[:] = token_ids
    mmap_array.flush()

    print("--- Data Preparation Complete ---")
    print(f"Your tokenized data is ready at: {args.output_npy}")

if __name__ == '__main__':
    main()