from dataclasses import dataclass, field
import heapq

import torch

from data.requests.prefill_request import PrefillRequest

@dataclass(order=True)
class DecodeRequest:
    arrival_time: float
    request_id: str = field(compare=False)
    kv_cache: torch.Tensor = field(compare=False)
    max_new_tokens: int = field(default=128, compare=False)
    text: str = field(default="", compare=False)

    @staticmethod
    def from_prefill_request(prefill_request: PrefillRequest, kv_cache: torch.Tensor):
        return DecodeRequest(prefill_request.arrival_time, prefill_request.request_id, kv_cache)

class DecodeRequestManager:
    def __init__(self):
        self.pending_requests: list[DecodeRequest] = []

    def add(self, request: DecodeRequest):
        heapq.heappush(self.pending_requests, request)

    def pop(self):
        return heapq.heappop(self.pending_requests) if self.pending_requests else None

    def __len__(self):
        return len(self.pending_requests)