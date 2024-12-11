import asyncio
import ray
from responses import stop
import torch
from typing import Optional

from config.actors.prefiller_config import PrefillerConfig
from config.models.mock_model.simple_attention_config import SimpleAttentionConfig
from data.requests.decode_request import DecodeRequest
from data.requests.prefill_request import PrefillRequest, PrefillRequestManager
from models.mock_model.simple_tokenizer import SimpleTokenizer
from models.mock_model.simple_attention_model import SimpleAttentionModel

@ray.remote
class PrefillActor:
    def __init__(self, config: PrefillerConfig, ref: "ray.actor.ActorHandle", model: str, tokenizer: str, controller: "ray.actor.ActorHandle"):
        self.ref = ref
        self.controller = controller

        self.tokenizer = SimpleTokenizer()
        self.model = SimpleAttentionModel(SimpleAttentionConfig())

        self.current_request: Optional[PrefillRequest] = None
        self.running = True
        
        self.pending_requests: PrefillRequestManager = PrefillRequestManager()

        self.loop = asyncio.get_event_loop()
        self.process_task = None
    
    def add_request(self, request: PrefillRequest):
        self.pending_requests.add(request)

    def _step(self):
        if not self.current_request:
            request = self.pending_requests.pop()
            if request:
                self.current_request = request

                tokens = self.tokenizer.encode(request.prompt)
                _, (K, V) = self.model.forward(tokens)
                kv_cache = torch.cat([K, V], dim=0)

                request.decoder.add_request(DecodeRequest.from_prefill_request(request, kv_cache))
                
                self.current_request = None
            else :
                print("prefiller idling")

    async def _main_loop(self):
        while self.running:
            try:
                self._step()
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Error in prefiller main loop: {e}")
                stop()

    def start(self):
        if not self.process_task:
            self.running = True
            self.process_task = asyncio.create_task(self._main_loop())

    def stop(self):
        self.running = False
        if self.process_task:
            self.process_task.cancel()
            self.process_task = None
        self.controller.unregister.remote(self.ref)
            
    def ping(self) -> bool:
        return True
