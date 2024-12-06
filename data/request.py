from dataclasses import dataclass, field
from uuid import uuid4

@dataclass
class Request:
    prompt: str
    sender: str
    uuid: str = field(default_factory=lambda: str(uuid4())) # do we need this?