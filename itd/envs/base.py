"""Environment interface the task solver and the ITD lifecycle rely on."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..model import Task


@dataclass
class StepResult:
    observation: str
    reward: float
    done: bool
    success: bool = False


class Environment(ABC):
    name: str
    context: str            # description of the environment and its actions ({context} in the prompts)
    max_steps: int          # horizon limit of one attempt

    @abstractmethod
    def tasks(self, split: str) -> list[Task]: ...

    @abstractmethod
    def reset(self, task: Task) -> str:
        """Start an attempt at `task`; return the first observation."""

    @abstractmethod
    def step(self, action: str) -> StepResult: ...

    @abstractmethod
    def progress(self) -> float:
        """m/M: fraction of subtasks completed by the current attempt. Only
        used to assign trace quality; never shown to the task solver."""

    def task_meta(self, task: Task) -> dict:
        """Extra per-task attributes copied into the result records."""
        return {}

    def close(self) -> None:
        pass


def make_env(benchmark: str, **kw) -> Environment:
    if benchmark == "alfworld":
        from .alfworld import ALFWorldEnv
        return ALFWorldEnv(**kw)
    if benchmark == "webarena":
        from .webarena import WebArenaEnv
        return WebArenaEnv(**kw)
    raise ValueError(f"unknown benchmark: {benchmark}")
