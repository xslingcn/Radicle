import asyncio
import ray
import torch
from queue import Queue
from models.simple_attention_model import SimpleAttentionModel

@ray.remote
class DecodingActor:
    pass