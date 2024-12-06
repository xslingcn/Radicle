from altair import Optional
import ray
import asyncio

import torch
from data.request import Request
from models.simple_attention_model import SimpleAttentionModel
from ray.dag import InputNode
from ray.experimental.compiled_dag_ref import CompiledDAGRef


@ray.remote
class PrefillActor:
    def __init__(self, model: str, driver: "ray.actor.ActorHandle"):
        self.model = SimpleAttentionModel()

        assign_dag = driver.assign.bind(InputNode())
        self.assign_dag = assign_dag.experimental_compile()

        insert_dag = driver.insert.bind(InputNode())
        self.insert_dag = insert_dag.experimental_compile()

        self.pending_requests: list[tuple[CompiledDAGRef, Request]] = []

        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self.request_job())
        self.loop.create_task(self.process_job())

        self.executing = False

    async def add_request(self, request: Request):
        pass

    async def step(self):
        if self.pending_requests and self.executing == False:
            self.executing = True
            job_ref, request = self.pending_requests.pop(0)
            output = self.model.forward(request.prompt)

            self.insert_dag.execute(job_ref)
                
