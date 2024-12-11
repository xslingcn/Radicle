from dataclasses import dataclass, field
import heapq

@dataclass(order=True)
class UserRequest:
    arrival_time: float
    request_id: str = field(compare=False)
    prompt: str = field(compare=False)

class UserRequestManager:
    def __init__(self):
        self.pending_requests: list[UserRequest] = []

    def add(self, request: UserRequest):
        heapq.heappush(self.pending_requests, request)

    def pop(self):
        return heapq.heappop(self.pending_requests) if self.pending_requests else None

    def __len__(self):
        return len(self.pending_requests)