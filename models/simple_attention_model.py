import torch

D_MODEL = 512

class SimpleAttentionModel:
    def __init__(self):
        self.W_Q = torch.randn(D_MODEL, D_MODEL)
        self.W_K = torch.randn(D_MODEL, D_MODEL)
        self.W_V = torch.randn(D_MODEL, D_MODEL)
        self.W_O = torch.randn(D_MODEL, D_MODEL)

    def forward(self, x, kv_cache=None):
        Q = x @ self.W_Q
        if kv_cache is None:
            K = x @ self.W_K
            V = x @ self.W_V
        else:
            K, V = kv_cache

        scores = Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(D_MODEL))
        attn_weights = torch.softmax(scores, dim=-1)
        output = attn_weights @ V
        output = output @ self.W_O

        return output, (K, V)