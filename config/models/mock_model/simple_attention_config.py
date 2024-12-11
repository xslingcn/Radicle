from dataclasses import dataclass
from config.models.base_config import BaseModelConfig


@dataclass
class SimpleAttentionConfig(BaseModelConfig):
    d_model: int = 512
