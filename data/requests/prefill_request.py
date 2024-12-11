from dataclasses import dataclass, field
import heapq

import ray

from data.requests.user_request import UserRequest


@dataclass(order=True)
class PrefillRequest:
    arrival_time: float
    request_id: str = field(compare=False)
    prompt: str = field(compare=False)
    decoder: "ray.actor.ActorHandle" = field(compare=False)

    @staticmethod
    def from_user_request(user_request: UserRequest, decoder: "ray.actor.ActorHandle"):
        return PrefillRequest(
            user_request.arrival_time,
            user_request.request_id,
            user_request.prompt,
            decoder,
        )


class PrefillRequestManager:
    def __init__(self):
        self.pending_requests: list[PrefillRequest] = []

    def add(self, request: PrefillRequest):
        heapq.heappush(self.pending_requests, request)

    def pop(self):
        return heapq.heappop(self.pending_requests) if self.pending_requests else None

    def __len__(self):
        return len(self.pending_requests)
