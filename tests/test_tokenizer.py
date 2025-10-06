# test_tokenizer.py (Corrected version)
from tokenizer import Tokenizer
import os

def get_test_documents(file_path: str, num_docs: int = 2) -> str:
    """
    Reads the first few documents from a text file for testing.
    This assumes documents are separated by '<|endoftext|>'.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    docs = content.split('<|endoftext|>')
    test_text = '<|endoftext|>'.join(docs[:num_docs]) + '<|endoftext|>'
    return test_text

# --- Pytest discoverable test function ---
def test_tokenizer_round_trip():
    """
    Tests that encoding and then decoding a text returns the original text.
    """
    # --- Configuration ---
    VOCAB_FILEPATH = "vocab.json"
    MERGES_FILEPATH = "merges.txt"
    SPECIAL_TOKENS = ["<|endoftext|>"]
    TEST_DATA_PATH = "/scratch.global/ruan0073/llm_scratch/data/TinyStoriesV2-GPT4-train.txt"

    # 1. Check if required files exist before running the test
    assert os.path.exists(VOCAB_FILEPATH), f"Vocabulary file not found at {VOCAB_FILEPATH}. Run train_tokenizer.py first."
    assert os.path.exists(MERGES_FILEPATH), f"Merges file not found at {MERGES_FILEPATH}. Run train_tokenizer.py first."

    # 2. Instantiate the tokenizer from saved files
    tokenizer = Tokenizer.from_files(
        vocab_filepath=VOCAB_FILEPATH,
        merges_filepath=MERGES_FILEPATH,
        special_tokens=SPECIAL_TOKENS
    )

    # 3. Get sample text that was NOT used in training
    original_text = get_test_documents(TEST_DATA_PATH, num_docs=2)
    
    # 4. Perform the round-trip: encode then decode
    encoded_ids = tokenizer.encode(original_text)
    decoded_text = tokenizer.decode(encoded_ids)

    # 5. Verify the result using an assert statement
    assert original_text == decoded_text, "Failure: Decoded text does NOT match the original text."

# You can keep this block if you still want to run the file directly as a script
if __name__ == '__main__':
    print("--- Running BPE Tokenizer Test as a Script ---")
    test_tokenizer_round_trip()
    print("\n--- Verification ---")
    print("✅ Success: Script ran without assertion errors.")