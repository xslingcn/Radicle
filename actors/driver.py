import ray
from ray.dag import InputNode
from ray.experimental.compiled_dag_ref import CompiledDAGRef
from typing import Optional
import torch
from data.pd_actor import PDActor
from data.request import Request
from data.role import Role


@ray.remote
class Driver:
    def __init__(self):
        self.actors_by_role: dict[Role, dict["ray.actor.ActorHandle", PDActor]] = {
            Role.PREFILLER: {},
            Role.DECODER: {},
        }
        self.actor_refs: set["ray.actor.ActorHandle"] = set()
        self.kv_cache: set[torch.Tensor] = set()
        self.pending_requests: list[Request] = []

    def _get_actor(self, actor_ref: "ray.actor.ActorHandle") -> PDActor:
        if actor_ref not in self.actor_refs:
            raise ValueError("Actor not found in registry")

        actor = None
        for role_actors in self.actors_by_role.values():
            if actor_ref in role_actors:
                actor = role_actors[actor_ref]
                break

        if actor is None:
            raise ValueError("Actor not found in registry")

        return actor

    def register(self, actor_ref: "ray.actor.ActorHandle") -> Role:
        prefiller_count = len(self.actors_by_role[Role.PREFILLER])
        decoder_count = len(self.actors_by_role[Role.DECODER])

        prefiller_workload = sum(
            actor.workload() for actor in self.actors_by_role[Role.PREFILLER].values()
        ) / (prefiller_count or 1)
        decoder_workload = sum(
            actor.workload() for actor in self.actors_by_role[Role.DECODER].values()
        ) / (decoder_count or 1)

        if prefiller_count == 0 or (
            prefiller_workload > decoder_workload and prefiller_count <= decoder_count
        ):
            role = Role.PREFILLER
        else:
            role = Role.DECODER

        dag = actor_ref.add_request.bind(InputNode())
        compiled_dag = dag.experimental_compile()

        pd_actor = PDActor(dag=compiled_dag, jobs=[], role=role)
        self.actors_by_role[role][actor_ref] = pd_actor
        self.actor_refs.add(actor_ref)
        return role

    def unregister(self, actor_ref: "ray.actor.ActorHandle") -> None:
        if actor_ref not in self.actor_refs:
            return

        for role_actors in self.actors_by_role.values():
            if actor_ref in role_actors:
                del role_actors[actor_ref]
        self.actor_refs.remove(actor_ref)

    def generate(self,sender:str, prompt: str) -> None:
        self.pending_requests.append(Request(sender=sender, prompt=prompt))

    def assign(self, actor_ref: "ray.actor.ActorHandle") -> Optional[CompiledDAGRef]:
        if not self.pending_requests:
            return None

        request = self.pending_requests.pop(0)
        actor = self._get_actor(actor_ref)
        if not actor.role == Role.PREFILLER:
            raise ValueError("Only prefillers can request jobs")

        job_ref = actor.dag.execute(request)
        if not isinstance(job_ref, CompiledDAGRef):
            raise RuntimeError("Invalid job ref type")
        actor.jobs.append(job_ref)

        return job_ref

    def insert(
        self, job_ref: ray.ObjectRef, actor_ref: "ray.actor.ActorHandle"
    ) -> None:
        actor = self._get_actor(actor_ref)
        if actor.role != Role.PREFILLER:
            raise ValueError("Only prefillers can insert KV cache")
        if job_ref not in actor.jobs:
            raise ValueError("Job not found in actor's jobs")

        self.kv_cache.add(job_ref.get())
        actor.jobs.remove(job_ref)

    def drop_select(
        self, actor_ref: "ray.actor.ActorHandle"
    ) -> Optional[CompiledDAGRef]:
        actor = self._get_actor(actor_ref)
        if actor.role != Role.DECODER:
            raise ValueError("Only decoders can drop select")

        if not self.kv_cache:
            return None

        selected_kv = self.kv_cache.pop()
        job_ref = actor.dag.execute(selected_kv)
        if not isinstance(job_ref, CompiledDAGRef):
            raise RuntimeError("Invalid job ref type")
        actor.jobs.append(job_ref)
        return job_ref

    def output(
        self, job_ref: ray.ObjectRef, actor_ref: "ray.actor.ActorHandle"
    ) -> None:
        actor = self._get_actor(actor_ref)
        if actor.role != Role.DECODER:
            raise ValueError("Only decoders can output")
        if job_ref not in actor.jobs:
            raise ValueError("Job not found in actor's jobs")

        actor.jobs.remove(job_ref)
        generated = job_ref.get()
        print(generated)
