from dataclasses import dataclass


@dataclass
class StreamState:
    text: str = ""
    is_complete: bool = False
    total_tokens: int = 0
