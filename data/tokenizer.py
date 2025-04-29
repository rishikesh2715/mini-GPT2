import tiktoken

class GPT2Tokenizer:
    def __init__(self, model_name="gpt2"):
        self.encoder = tiktoken.get_encoding(model_name)

    def encode(self, text):
        return self.encoder.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens):
        return self.encoder.decode(tokens)
