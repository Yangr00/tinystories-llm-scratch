import os
import regex as re
from typing import BinaryIO, List, Dict, Tuple
from collections import Counter
import multiprocessing
import json
import base64

# Regex pattern for pre-tokenization, as specified in the prompt.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# --- Helper functions for Step 2: Merging ---

def get_stats(splits: Dict[Tuple[int, ...], int]) -> Counter:
    """
    Computes the frequency of adjacent pairs of tokens.
    'splits' is a dictionary mapping a tuple of token IDs (a word) to its frequency.
    """
    pair_counts = Counter()
    for word_tokens, freq in splits.items():
        for i in range(len(word_tokens) - 1):
            pair = (word_tokens[i], word_tokens[i+1])
            pair_counts[pair] += freq
    return pair_counts

def merge(
    pair: Tuple[int, int],
    new_token_id: int,
    splits: Dict[Tuple[int, ...], int]
) -> None:
    """
    Performs a merge operation by replacing all occurrences of 'pair' with 'new_token_id'.
    This version directly modifies the splits dictionary for the next iteration.
    """
    p1, p2 = pair
    new_splits = {}
    for word_tokens, freq in splits.items():
        if len(word_tokens) < 2:
            new_splits[word_tokens] = freq
            continue

        new_word_tokens = []
        i = 0
        while i < len(word_tokens):
            if i < len(word_tokens) - 1 and word_tokens[i] == p1 and word_tokens[i+1] == p2:
                new_word_tokens.append(new_token_id)
                i += 2
            else:
                new_word_tokens.append(word_tokens[i])
                i += 1
        
        new_splits[tuple(new_word_tokens)] = freq
    
    splits.clear()
    splits.update(new_splits)


# --- Functions for Step 1: Pre-tokenization ---

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size
    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))

def _worker_pre_tokenize(args: Tuple) -> Counter:
    """
    Worker function for parallel pre-tokenization.
    """
    input_path, start_byte, end_byte, special_token_str, compiled_regex = args
    special_token_bytes = special_token_str.encode("utf-8")
    word_counts = Counter()
    with open(input_path, "rb") as f:
        f.seek(start_byte)
        chunk_bytes = f.read(end_byte - start_byte)
    text_segments = chunk_bytes.split(special_token_bytes)
    if len(text_segments) > 1:
        word_counts[special_token_str] = len(text_segments) - 1
    for segment in text_segments:
        text = segment.decode("utf-8", errors="ignore")
        if text:
            pre_tokens = compiled_regex.findall(text)
            word_counts.update(pre_tokens)
    return word_counts


# --- Main BPE Training Function ---

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Trains a BPE tokenizer from a text file, performing both pre-tokenization and merging.
    """
    if not special_tokens:
        raise ValueError("At least one special token must be provided.")
    split_special_token_str = special_tokens[0]
    split_special_token_bytes = split_special_token_str.encode("utf-8")
    num_processes = 2
    compiled_pat = re.compile(PAT)

    # --- STEP 1: PARALLEL PRE-TOKENIZATION ---
    print(f"--- Step 1: Pre-tokenization ---")
    print(f"Starting with {num_processes} processes...")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_special_token_bytes)
    if len(boundaries) - 1 < num_processes:
        num_processes = len(boundaries) - 1
    chunk_ranges = list(zip(boundaries[:-1], boundaries[1:]))
    worker_args = [(input_path, s, e, split_special_token_str, compiled_pat) for s, e in chunk_ranges]
    with multiprocessing.Pool(num_processes) as pool:
        list_of_counters = pool.map(_worker_pre_tokenize, worker_args)
    total_word_counts = Counter()
    for counter in list_of_counters:
        total_word_counts.update(counter)
    print(f"Found {len(total_word_counts):,} unique pre-tokens.")

    # --- STEP 2: BPE MERGING ---
    print(f"\n--- Step 2: Merging ---")
    
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token_str in enumerate(special_tokens):
        vocab[256 + i] = token_str.encode("utf-8")
    
    splits = {
        tuple(word.encode("utf-8")): freq
        for word, freq in total_word_counts.items()
    }

    merges = []
    
    num_merges_needed = vocab_size - len(vocab)
    print(f"Targeting {num_merges_needed} merges to reach vocab size of {vocab_size}.")

    for i in range(num_merges_needed):
        pair_stats = get_stats(splits)
        if not pair_stats:
            break
        
        best_pair = max(pair_stats, key=lambda p: (pair_stats[p], p))
        
        new_token_id = len(vocab)
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        
        merge(best_pair, new_token_id, splits)

        if (i + 1) % 50 == 0:
            print(f"Merge {i + 1}/{num_merges_needed} completed.")
            
    return vocab, merges

if __name__ == '__main__':
    DATASET_PATH = "/scratch.global/ruan0073/llm_scratch/data/TinyStoriesV2-GPT4-train.txt"
    VOCAB_SIZE = 10000
    SPECIAL_TOKENS = ["<|endoftext|>"]
    
    # Define filepaths for the output
    VOCAB_FILEPATH = "vocab.json"
    MERGES_FILEPATH = "merges.txt"

    print("--- Starting BPE Tokenizer Training ---")
    
    final_vocab, final_merges = train_bpe(
        input_path=DATASET_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS
    )
    
    print("\n--- Training Complete ---")
    
    # --- SAVE THE ARTIFACTS ---
    print(f"Saving vocabulary to {VOCAB_FILEPATH}")
    # We need to serialize the bytes values in the vocab.
    # A common way is to store them as a list of integers.
    serializable_vocab = {token_id: list(byte_val) for token_id, byte_val in final_vocab.items()}
    with open(VOCAB_FILEPATH, "w") as f:
        json.dump(serializable_vocab, f)

    print(f"Saving merges to {MERGES_FILEPATH}")
    with open(MERGES_FILEPATH, "w") as f:
        for p1, p2 in final_merges:
            # Encode bytes to base64 ascii strings. This is a robust way
            # to save binary data in a text file without delimiter issues.
            p1_b64 = base64.b64encode(p1).decode('ascii')
            p2_b64 = base64.b64encode(p2).decode('ascii')
            f.write(f"{p1_b64} {p2_b64}\n")

    print("\nRun complete. Artifacts saved.")