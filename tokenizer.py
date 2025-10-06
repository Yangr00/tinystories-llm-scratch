import regex as re
import json
from typing import List, Dict, Tuple, Optional, Iterable, Iterator
import base64

# Use the same regex pattern from training
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None
    ):
        self.vocab = vocab
        self.merges = {pair: i for i, pair in enumerate(merges)} # Faster lookups
        
        # Invert the vocab for encoding
        self.byte_to_token_id = {v: k for k, v in self.vocab.items()}
        
        self.special_tokens = special_tokens or []
        if self.special_tokens:
            for token in self.special_tokens:
                if token.encode("utf-8") not in self.byte_to_token_id:
                    # Append new special tokens if they weren't in the training vocab
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token.encode("utf-8")
                    self.byte_to_token_id[token.encode("utf-8")] = new_id

        self.compiled_pat = re.compile(PAT)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None
    ) -> "Tokenizer":
        """
        Constructs a Tokenizer from saved vocabulary and merges files.
        """
        
        # Load vocabulary
        with open(vocab_filepath, 'r') as f:
            serializable_vocab = json.load(f)
        vocab = {int(k): bytes(v) for k, v in serializable_vocab.items()}

        # Load merges
        with open(merges_filepath, 'r') as f:
            merges_text = f.read().splitlines()
        
        merges = []
        for line in merges_text:
            # Decode the base64 strings back to bytes
            p1_b64, p2_b64 = line.split(' ')
            p1 = base64.b64decode(p1_b64)
            p2 = base64.b64decode(p2_b64)
            merges.append((p1, p2))

        return cls(vocab, merges, special_tokens)

    def decode(self, ids: List[int]) -> str:
        """
        Decodes a sequence of token IDs back into a string.
        """
        all_bytes = b"".join(self.vocab.get(i, b'') for i in ids)
        # The 'replace' error handling is crucial as per the requirements
        return all_bytes.decode("utf-8", errors="replace")

    def _encode_chunk(self, text_bytes: bytes) -> List[int]:
        """Helper to encode a single chunk of text after pre-tokenization."""
        tokens = list(text_bytes) # Start with a list of byte integers
        
        while len(tokens) >= 2:
            # Find the best merge available in this chunk
            # The rank is the merge's position in the ordered list
            stats = {}
            for i in range(len(tokens) - 1):
                pair = (self.vocab[tokens[i]], self.vocab[tokens[i+1]])
                if pair in self.merges:
                    stats[i] = self.merges[pair]
            
            if not stats:
                break # No more merges possible in this chunk

            best_pair_idx = min(stats, key=stats.get)
            
            # Perform the merge
            p1_bytes = self.vocab[tokens[best_pair_idx]]
            p2_bytes = self.vocab[tokens[best_pair_idx+1]]
            merged_bytes = p1_bytes + p2_bytes
            new_token_id = self.byte_to_token_id[merged_bytes]

            tokens = tokens[:best_pair_idx] + [new_token_id] + tokens[best_pair_idx+2:]
        
        return tokens


    def encode(self, text: str) -> List[int]:
        """
        Encodes an input string into a sequence of token IDs.
        """
        all_ids = []
        # Handle special tokens by splitting the text
        # This prevents merges across special token boundaries
        special_pattern = "|".join(re.escape(token) for token in self.special_tokens) if self.special_tokens else ""
        
        if special_pattern:
            chunks = re.split(f'({special_pattern})', text)
        else:
            chunks = [text]

        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens:
                all_ids.append(self.byte_to_token_id[chunk.encode("utf-8")])
            else:
                # Regular pre-tokenization and merging
                pre_tokens = self.compiled_pat.findall(chunk)
                for pre_token in pre_tokens:
                    all_ids.extend(self._encode_chunk(pre_token.encode("utf-8")))
        
        return all_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encodes an iterable of strings, yielding token IDs.
        """
        for text_chunk in iterable:
            yield from self.encode(text_chunk)