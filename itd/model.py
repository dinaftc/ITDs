"""The data model: Task, Trace, Step, Experience."""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field

SUCCESS = "success"
NEAR_SUCCESS = "near-success"
FAILURE = "failure"
CATEGORIES = (SUCCESS, NEAR_SUCCESS, FAILURE)

EXAMPLES = "examples"
RULES = "rules"
REFLECTIONS = "reflections"
COMBINED = "combined"
STRATEGIES = (EXAMPLES, RULES, REFLECTIONS, COMBINED)


@dataclass(frozen=True)
class Task:
    """A task the agent is asked to do, and the template it instantiates."""

    task_id: str
    task_family: str
    template_desc: str
    task_desc: str


@dataclass
class Step:
    """One turn of a trace; the trace it belongs to is implied by the owner."""

    step_index: int
    observation: str
    action: str


@dataclass
class Trace:
    """One attempt at a task, with the attributes the store indexes on.

    `progress` (m/M, as reported by the environment) and `task_desc` are not
    part of the relation: progress is only needed to assign the quality
    category, and task_desc is joined from Task when traces are retrieved.
    """

    task_id: str
    task_family: str
    template_desc: str
    actor: str
    label: str
    reward: float
    horizon: int
    steps: list[Step] = field(default_factory=list)
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    quality_category: str | None = None
    quality_score: float | None = None
    progress: float | None = None
    task_desc: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Trace":
        d = dict(d)
        d["steps"] = [Step(**s) for s in d.get("steps", [])]
        return cls(**d)


@dataclass
class Experience:
    """Reusable material derived from retrieved traces.

    `synthesizer` is "" for Examples, which needs no LLM to assemble.
    """

    task_family: str
    template_desc: str
    strategy: str
    synthesizer: str
    content: str
    trace_ids: list[str]
    timestamp: float = field(default_factory=time.time)
