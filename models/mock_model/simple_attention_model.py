import torch

from config.models.mock_model.simple_attention_config import SimpleAttentionConfig


class SimpleAttentionModel:
    def __init__(self, config: SimpleAttentionConfig):
        self.d_model = config.d_model

        self.W_Q = torch.randn(self.d_model, self.d_model)
        self.W_K = torch.randn(self.d_model, self.d_model)
        self.W_V = torch.randn(self.d_model, self.d_model)
        self.W_O = torch.randn(self.d_model, self.d_model)

    def forward(
        self, input_ids, kv_cache=None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = torch.randn(input_ids.size(0), self.d_model)
        Q = x @ self.W_Q
        if kv_cache is None:
            K = x @ self.W_K
            V = x @ self.W_V
        else:
            K, V = kv_cache

        scores = Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(self.d_model))
        attn_weights = torch.softmax(scores, dim=-1)
        output = attn_weights @ V
        output = output @ self.W_O

        return output, (K, V)
