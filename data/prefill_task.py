from dataclasses import dataclass
from typing import List
from ray.experimental.compiled_dag_ref import CompiledDAGRef

@dataclass
class PrefillTask:
    job: CompiledDAGRef
    prompt: str
