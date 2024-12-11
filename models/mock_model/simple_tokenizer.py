import random
import torch


class SimpleTokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.eos_token = 0

    def encode(self, text):
        token_length = len(text)
        tokens = [random.randint(0, self.vocab_size - 1) for _ in range(token_length)]
        return torch.tensor(tokens)
    
    def decode(self, tokens):
        return ''.join([str(token.item()) for token in tokens])