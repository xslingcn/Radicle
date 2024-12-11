import asyncio
import time
import ray
from typing import Optional
from config.actors.controller_config import ControllerConfig
from data.pd_actor import PDActor
from data.requests.prefill_request import PrefillRequest
from data.role import Role

from data.requests.user_request import UserRequest, UserRequestManager
from data.stram_state import StreamState

@ray.remote
class Controller:
    def __init__(self, config: ControllerConfig):
        self.request_timeout = config.request_timeout

        self.prefiller: dict["ray.actor.ActorHandle", PDActor] = {}
        self.decoder: dict["ray.actor.ActorHandle", PDActor] = {}
        self.pending_requests: UserRequestManager = UserRequestManager()
        self.stream_states: dict[str, StreamState] = {}

        self.loop = asyncio.get_event_loop()
        self.process_task = None

    def register(self, actor_ref: "ray.actor.ActorHandle") -> Role:
        prefiller_count = len(self.prefiller)
        decoder_count = len(self.decoder)

        prefiller_workload = sum(actor.workload() for actor in self.prefiller.values())
        decoder_workload = sum(actor.workload() for actor in self.decoder.values())

        if prefiller_count == 0 or (
            prefiller_workload > decoder_workload and prefiller_count <= decoder_count
        ):
            role = Role.PREFILLER
        else:
            role = Role.DECODER

        pd_actor = PDActor(ref=actor_ref, requests=[], role=role)
        if role == Role.PREFILLER:
            self.prefiller[actor_ref] = pd_actor
        else:
            self.decoder[actor_ref] = pd_actor
        return role

    def unregister(self, actor_ref: "ray.actor.ActorHandle") -> None:
        if actor_ref in self.prefiller:
            if self.prefiller[actor_ref].workload() != 0:
                for request in self.prefiller[actor_ref].requests:
                    self.pending_requests.add(request)
            del self.prefiller[actor_ref]
            return

        if actor_ref in self.decoder:
            del self.decoder[actor_ref]
            return

    def add_request(self, request_id: str, prompt: str) -> None:
        arrival_time = time.time()
        self.pending_requests.add(UserRequest(arrival_time, request_id, prompt))

    def stream_token(self, request_id: str, token: str, is_complete: bool) -> None:
        """Handle streaming tokens from decoder"""
        state = self.stream_states.get(request_id, StreamState())
        
        if token:
            state.text += token
            print(f"Request {request_id}: {token}", end="", flush=True)
        
        if is_complete:
            state.is_complete = True
            print(f"\nRequest {request_id} complete: {state.text}")
            del self.stream_states[request_id]
            for decoder in self.decoder.values():
                decoder.requests = [r for r in decoder.requests if r.request_id != request_id]
        else:
            self.stream_states[request_id] = state

    def _step(self) -> None:
        while len(self.pending_requests) > 0:
            request = self.pending_requests.pop()
            if not request:
                break

            prefiller_ref, decoder_ref = self._get_least_loaded_actors()
            if not (prefiller_ref and decoder_ref):
                self.pending_requests.add(request)
                break

            self.stream_states[request.request_id] = StreamState()

            self.prefiller[prefiller_ref].requests.append(request)
            self.decoder[decoder_ref].requests.append(request)

            try:
                ray.get(
                    prefiller_ref.add_request.remote(
                        PrefillRequest.from_user_request(request, decoder_ref)
                    )
                )
            except (ray.exceptions.GetTimeoutError, ray.exceptions.RayActorError):
                self.unregister(prefiller_ref)
                self.pending_requests.add(request)
                continue

        current_time = time.time()
        for actor_type in [self.prefiller, self.decoder]:
            for actor_ref, actor in list(actor_type.items()):
                for request in actor.requests:
                    if current_time - request.arrival_time > self.request_timeout:
                        if not self._check_actor_health(actor_ref):
                            self.unregister(actor_ref)
                            if request.request_id in self.stream_states:
                                del self.stream_states[request.request_id]
                            break

    def _get_least_loaded_actors(
        self,
    ) -> tuple[Optional["ray.actor.ActorHandle"], Optional["ray.actor.ActorHandle"]]:
        if not self.prefiller or not self.decoder:
            return None, None

        prefiller_ref = min(self.prefiller.items(), key=lambda x: x[1].workload())[0]
        decoder_ref = min(self.decoder.items(), key=lambda x: x[1].workload())[0]
        return prefiller_ref, decoder_ref

    def _check_actor_health(self, actor_ref: "ray.actor.ActorHandle") -> bool:
        """Verify actor is responsive"""
        try:
            ray.get(actor_ref.ping.remote(), timeout=1.0)
            return True
        except (ray.exceptions.GetTimeoutError, ray.exceptions.RayActorError):
            return False

    async def _main_loop(self):
        while self.running:
            try:
                self._step()
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Error in controller main loop: {e}")

    def start(self):
        if not self.process_task:
            self.running = True
            self.process_task = asyncio.create_task(self._main_loop())

    def stop(self):
        self.running = False
        if self.process_task:
            self.process_task.cancel()
            self.process_task = None