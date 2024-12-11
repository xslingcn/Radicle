from dataclasses import dataclass, field
from .base_actor_config import BaseActorConfig


@dataclass
class DecoderConfig(BaseActorConfig):
    max_batch_size: int = 32
    batch_timeout: float = 0.01
