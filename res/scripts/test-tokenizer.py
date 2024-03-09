import sentencepiece as spm

# Load the tokenizer model
sp = spm.SentencePieceProcessor()
sp.load('tokenizer.model')

# Now you can use the `sp` object to encode and decode text.

# Example text
text = "This is an example sentence to be tokenized."

# Encode text into tokens
tokens = sp.encode_as_pieces(text)
print("Tokens:", tokens)

# Decode tokens back into text
decoded_text = sp.decode_pieces(tokens)
print("Decoded text:", decoded_text)
