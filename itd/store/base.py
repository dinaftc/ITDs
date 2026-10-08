"""Store interface shared by the in-memory, PostgreSQL and Qdrant instantiations.

The two ITD operators (Section 3.3) are `maintain` and `retrieve`; every
instantiation applies the same partitioning and trace ranking.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..model import CATEGORIES, Experience, Task, Trace

EXACT_FALLBACK = "exact+fallback"   # ITD retrieval: exact template, then similarity fallback
SIMILARITY_ONLY = "similarity"      # alternative evaluated in Section 4.8


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
        """Maintain(D, τ, k), Eq. (5): keep the top-k of P_{tpl(τ),c}(D) ∪ {τ}.

        `trace` must already carry its quality category and score, and its task
        must have been registered. Returns True iff τ is retained; in that case
        cached experiences of the template are invalidated.
        """

    @abstractmethod
    def retrieve(self, task: Task, category: str, n: int, *, exclude_task_id: str | None = None,
                 mode: str = EXACT_FALLBACK, query_vec: np.ndarray | None = None,
                 with_steps: bool = True) -> list[Trace]:
        """Retrieve(t, D, c, n), Eq. (6): at most n traces of category c.

        Traces of t's own task template come first. Remaining positions are
        filled from other templates of the same task family, ranked by cosine
        similarity between the template embedding and t's description; traces
        within a template are ranked by quality score. `exclude_task_id`
        implements leave-one-out evaluation.
        """

    def retrieve_all(self, task: Task, n: int, *, categories=CATEGORIES, **kw) -> list[Trace]:
        """RetrieveAll(t, D, n), Eq. (7): at most n traces per quality category."""
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
