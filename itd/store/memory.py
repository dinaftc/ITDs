"""In-memory reference instantiation. It defines the operator semantics the
database instantiations are tested against, and is enough to run the
task-success experiments on a single machine."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..model import Experience, Task, Trace
from ..quality import rank_key
from .base import EXACT_FALLBACK, ITDStore


class MemoryStore(ITDStore):
    def __init__(self, embedder):
        super().__init__(embedder)
        self.tasks: dict[str, Task] = {}
        self.templates: dict[str, dict[str, np.ndarray]] = defaultdict(dict)   # family -> template -> vec
        self.partitions: dict[tuple[str, str, str], list[Trace]] = defaultdict(list)
        self.experiences: dict[tuple[str, str, str, str], Experience] = {}

    def add_task(self, task: Task) -> None:
        self.tasks[task.task_id] = task
        if task.template_desc not in self.templates[task.task_family]:
            self.templates[task.task_family][task.template_desc] = self.embedder.encode([task.template_desc])[0]

    def maintain(self, trace: Trace, k: int) -> bool:
        key = (trace.task_family, trace.template_desc, trace.quality_category)
        # sorted() is stable: on a full tie the incumbent stays.
        kept = sorted(self.partitions[key] + [trace], key=rank_key)[:k]
        self.partitions[key] = kept
        retained = any(t is trace for t in kept)
        if retained:
            for ek in [ek for ek in self.experiences if ek[:2] == key[:2]]:
                del self.experiences[ek]
        return retained

    def _partition(self, family, template, category, exclude_task_id):
        return [t for t in self.partitions.get((family, template, category), ()) if t.task_id != exclude_task_id]

    def retrieve(self, task, category, n, *, exclude_task_id=None, mode=EXACT_FALLBACK, query_vec=None,
                 with_steps=True):
        out: list[Trace] = []
        if mode == EXACT_FALLBACK:
            out = self._partition(task.task_family, task.template_desc, category, exclude_task_id)[:n]
        if len(out) < n:
            q = self.embed_query(task) if query_vec is None else query_vec
            candidates = [(float(q @ v), tpl) for tpl, v in self.templates[task.task_family].items()
                          if not (mode == EXACT_FALLBACK and tpl == task.template_desc)]
            for _, tpl in sorted(candidates, key=lambda c: -c[0]):
                out.extend(self._partition(task.task_family, tpl, category, exclude_task_id)[: n - len(out)])
                if len(out) >= n:
                    break
        for t in out:
            t.task_desc = self.tasks[t.task_id].task_desc
        return out

    def cache_lookup(self, task, strategy, synthesizer):
        return self.experiences.get((task.task_family, task.template_desc, strategy, synthesizer))

    def cache_put(self, experience):
        e = experience
        self.experiences[(e.task_family, e.template_desc, e.strategy, e.synthesizer)] = e

    def has_task_traces(self, task_id):
        return any(t.task_id == task_id for part in self.partitions.values() for t in part)

    def count_traces(self):
        return sum(len(p) for p in self.partitions.values())
