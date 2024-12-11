import asyncio
import time
import ray
import torch
from typing import Dict, List
from collections import defaultdict

from config.actors.decoder_config import DecoderConfig
from config.models.mock_model.simple_attention_config import SimpleAttentionConfig
from data.requests.decode_request import DecodeRequest, DecodeRequestManager
from data.stram_state import StreamState
from models.mock_model.simple_attention_model import SimpleAttentionModel
from models.mock_model.simple_tokenizer import SimpleTokenizer


@ray.remote
class DecodingActor:
    def __init__(
        self,
        config: DecoderConfig,
        model: str,
        tokenizer: str,
        controller: "ray.actor.ActorHandle",
    ):
        self.ref = None
        self.controller = controller

        self.tokenizer = SimpleTokenizer()
        self.model = SimpleAttentionModel(SimpleAttentionConfig())

        self.pending_requests = DecodeRequestManager()
        self.active_batches: Dict[int, List[DecodeRequest]] = defaultdict(list)
        self.kv_caches: Dict[str, torch.Tensor] = {}

        self.max_batch_size = config.max_batch_size
        self.batch_timeout = config.batch_timeout

        self.stream_states: Dict[str, StreamState] = {}

        self.running = True

        self.loop = asyncio.get_event_loop()
        self.process_task = None

    def set_ref(self, ref: "ray.actor.ActorHandle"):
        self.ref = ref

    def add_request(self, request: DecodeRequest):
        self.pending_requests.add(request)
        self.kv_caches[request.request_id] = request.kv_cache

    def _step(self):
        if len(self.pending_requests) == 0:
            return

        current_time = time.time()
        batch_id = int(current_time * 1000)

        batch: List[DecodeRequest] = []
        while len(batch) < self.max_batch_size and len(self.pending_requests) > 0:
            request = self.pending_requests.pop()
            if not request:
                break

            if request.request_id not in self.kv_caches:
                continue

            batch.append(request)

        if not batch:
            return

        try:
            batch_kv = torch.stack([self.kv_caches[req.request_id] for req in batch])

            outputs, new_kv = self.model.forward(batch_kv)

            for i, request in enumerate(batch):
                self.kv_caches[request.request_id] = new_kv[i]

                stream_state = self.stream_states.get(request.request_id, StreamState())

                next_token = outputs[i].argmax()
                next_text = self.tokenizer.decode([next_token])

                stream_state.text += next_text
                stream_state.total_tokens += 1
                self.stream_states[request.request_id] = stream_state

                self.controller.stream_token.remote(
                    request.request_id, next_text, is_complete=False
                )

                is_complete = (
                    next_token == self.tokenizer.eos_token
                    or stream_state.total_tokens >= request.max_new_tokens
                )

                if is_complete:
                    self.controller.stream_token.remote(request.request_id, "", True)
                    del self.kv_caches[request.request_id]
                    del self.stream_states[request.request_id]
                else:
                    self.pending_requests.add(request)

        except Exception as e:
            print(f"Error processing batch {batch_id}: {e}")
            self.controller.unregister.remote(self.ref)

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

    async def _main_loop(self):
        while self.running:
            try:
                self._step()
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Error in decoder {self.ref} main loop: {e}")
                self.stop()

    def ping(self) -> bool:
        """Health check endpoint"""
        return True
