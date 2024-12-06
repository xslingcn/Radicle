from dataclasses import dataclass
from role import Role
from ray.dag.compiled_dag_node import CompiledDAG
from ray.experimental.compiled_dag_ref import CompiledDAGRef

@dataclass
class PDActor:
    dag: CompiledDAG
    jobs: list[CompiledDAGRef]
    role: Role
    
    def workload(self) -> int:
        return len(self.jobs)