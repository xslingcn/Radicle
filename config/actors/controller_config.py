from dataclasses import dataclass

from attr import field
from .base_actor_config import BaseActorConfig


@dataclass
class ControllerConfig(BaseActorConfig):
    request_timeout: float = 0.01
