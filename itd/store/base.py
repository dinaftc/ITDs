"""Store interface shared by the in-memory, PostgreSQL and Qdrant backends.

All three apply the same partitioning and ranking. A partition is one
(task template, quality category); the store holds at most 3k|T| traces.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..model import CATEGORIES, Experience, Task, Trace

EXACT_FALLBACK = "exact+fallback"   # exact template, similarity only on underfill
SIMILARITY_ONLY = "similarity"      # always rank templates by embedding


class ITDStore(ABC):
    def __init__(self, embedder):
        self.embedder = embedder

    def embed_query(self, task: Task) -> np.ndarray:
        return self.embedder.encode([task.task_desc])[0]

    @abstractmethod
    def add_task(self, task: Task) -> None:
        """Register a task and, if new, its task template (with its embedding)."""

    @abstractmethod
    def maintain(self, trace: Trace, k: int) -> bool:
        """Keep the k best of the partition plus the incoming trace, ranked by
        quality score with ties broken by shorter horizon.

        `trace` must already carry its quality category and score, and its task
        must be registered. Returns True if it is retained, in which case the
        template's cached experiences are invalidated.
        """

    @abstractmethod
    def retrieve(self, task: Task, category: str, n: int, *, exclude_task_id: str | None = None,
                 mode: str = EXACT_FALLBACK, query_vec: np.ndarray | None = None,
                 with_steps: bool = True) -> list[Trace]:
        """At most n traces of one quality category.

        The task's own template first, then other templates of the same family
        ranked by cosine similarity to the task description; within a template,
        by quality score. `exclude_task_id` gives leave-one-out evaluation.
        """

    def retrieve_all(self, task: Task, n: int, *, categories=CATEGORIES, **kw) -> list[Trace]:
        """At most n traces from each quality category, in category order."""
        if kw.get("query_vec") is None:
            kw["query_vec"] = self.embed_query(task)
        out: list[Trace] = []
        for c in categories:
            out.extend(self.retrieve(task, c, n, **kw))
        return out

    @abstractmethod
    def cache_lookup(self, task: Task, strategy: str, synthesizer: str) -> Experience | None: ...

    @abstractmethod
    def cache_put(self, experience: Experience) -> None: ...

    @abstractmethod
    def has_task_traces(self, task_id: str) -> bool: ...

    @abstractmethod
    def count_traces(self) -> int: ...

    def close(self) -> None:
        pass
