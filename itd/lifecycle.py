"""Retrieve, solve, maintain, for one task."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .model import CATEGORIES, SUCCESS, Experience, Task, Trace
from .quality import assign_quality
from .solver import TaskSolver
from .store import ITDStore
from .strategies import components, inject, materialize

OFFLINE, ONLINE, HYBRID = "offline", "online", "hybrid"
ON_DEMAND, FAILURE_TRIGGERED = "on-demand", "failure-triggered"


@dataclass
class Lifecycle:
    store: ITDStore
    solver: TaskSolver
    synthesizer: object | None            # LLM used for Rules / Reflections
    approach: str = OFFLINE               # maintenance approach m
    policy: str = ON_DEMAND               # retrieval policy p
    strategy: str = "rules"               # reuse strategy s
    k: int = 1                            # maintenance budget
    n: int = 1                            # retrieval budget
    theta: float = 0.5                    # near-success threshold
    categories: tuple = CATEGORIES        # quality categories available to RetrieveAll
    leave_one_out: bool = False           # never reuse traces of the task being evaluated
    max_obs_chars: int = 300
    timing: dict = field(default_factory=lambda: {"retrieve_s": 0.0, "synthesis_s": 0.0})

    def maintain(self, task: Task, trace: Trace) -> bool:
        assign_quality(trace, self.theta)
        self.store.add_task(task)
        return self.store.maintain(trace, self.k)

    def experience(self, task: Task) -> list[Experience]:
        """The cached experience for this template, or build one and cache it."""
        exclude = task.task_id if self.leave_one_out else None
        # A cached experience may derive from the held-out task's own traces.
        use_cache = not (exclude and self.store.has_task_traces(exclude))
        out, traces = [], None
        for s in components(self.strategy):
            synth_name = "" if s == "examples" else self.synthesizer.name
            e = self.store.cache_lookup(task, s, synth_name) if use_cache else None
            if e is None:
                if traces is None:
                    start = time.perf_counter()
                    traces = self.store.retrieve_all(task, self.n, categories=self.categories,
                                                     exclude_task_id=exclude)
                    self.timing["retrieve_s"] += time.perf_counter() - start
                if not traces:
                    return []
                start = time.perf_counter()
                e = materialize(traces, s, self.synthesizer, task, self.solver.env.context, self.max_obs_chars)
                self.timing["synthesis_s"] += time.perf_counter() - start
                if use_cache:
                    self.store.cache_put(e)
            out.append(e)
        return out

    def run(self, task: Task) -> tuple[Trace, dict]:
        """Returns the completed trace and bookkeeping for the result record."""
        info = {"attempts": 1, "first_attempt_success": None, "experience_traces": 0}
        if self.policy == FAILURE_TRIGGERED:
            first = self.solver.complete(task)
            info["first_attempt_success"] = first.label == SUCCESS
            if first.label == SUCCESS:
                if self.approach != OFFLINE:
                    self.maintain(task, first)
                return first, info
            # Never maintained, so the retry sees only what was retained
            # before this task.
            info["attempts"] = 2
        experiences = self.experience(task)
        info["experience_traces"] = len(experiences[0].trace_ids) if experiences else 0
        trace = self.solver.complete(task, inject(experiences) if experiences else None)
        if self.approach != OFFLINE:
            self.maintain(task, trace)
        return trace, info
