import numpy as np
from sentencepiece import SentencePieceProcessor

# Load the tokenizer
tokenizer_path = './tokenizer.model'
tokenizer = SentencePieceProcessor(model_file=tokenizer_path)

# Function to read the binary file and convert to human-readable text
def read_and_decode(filename, tokenizer):
    with open(filename, 'rb') as f:
        # Read the binary data
        binary_data = f.read()
    # Convert from binary to int16 array
    token_ids = np.frombuffer(binary_data, dtype=np.int16)
    # Remove padding tokens
    token_ids = token_ids[token_ids != -100]
    # Decode token ids to text
    text = tokenizer.decode(token_ids.tolist())
    return text

# Function to convert text to JSONL format (if possible)
def text_to_jsonl(text):
    # Assuming the text is a simple string, we can create a basic JSONL object
    # If the original JSONL structure is known, this function can be modified accordingly
    jsonl_data = {"text": text}
    return jsonl_data

# Read and decode the binary file
decoded_text = read_and_decode('./instruct_dataset_minimal/data00.bin', tokenizer)

# Print the human-readable text
print("Decoded Text:")
print(decoded_text)

# Convert text to JSONL format
jsonl_data = text_to_jsonl(decoded_text)

Print the JSONL data
print("\nJSONL Data:")
print(jsonl_data)
