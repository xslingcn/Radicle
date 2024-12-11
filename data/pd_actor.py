from dataclasses import dataclass
from data.requests.user_request import UserRequest
from role import Role
import ray


@dataclass
class PDActor:
    ref: "ray.actor.ActorHandle"
    requests: list[UserRequest]
    role: Role

    def workload(self) -> int:
        return len(self.requests)
