from dataclasses import dataclass
from typing import Callable, Dict, List, Union


@dataclass
class Task:
    id: str
    run: Union[Callable[[], dict], Callable[[Dict], dict]]  # Support both signatures
    deps: List[str] = None


class DAG:
    def __init__(self, tasks: List[Task]):
        self.tasks = {t.id: t for t in tasks}
        self.context = {}  # Store task outputs for downstream tasks

    def execute(self):
        completed = set()
        while len(completed) < len(self.tasks):
            progressed = False
            for tid, t in self.tasks.items():
                if tid in completed:
                    continue
                if not t.deps or all(d in completed for d in (t.deps or [])):
                    # Try calling with context first, fall back to no-args
                    try:
                        manifest = t.run(self.context)
                    except TypeError:
                        # Function doesn't accept context argument
                        manifest = t.run()

                    print(f"[OK] Task {tid}: {manifest}")

                    # Store manifest in context for downstream tasks
                    self.context[tid] = manifest
                    completed.add(tid)
                    progressed = True
            if not progressed:
                raise RuntimeError("DAG deadlock: dependencies not satisfied")
