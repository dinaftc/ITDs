"""A tiny text environment for dry runs and tests.

Each task is a fixed sequence of three subtasks, e.g. take / heat / put, so
attempts end as success, near-success or failure like on the real benchmarks.
"""
from __future__ import annotations

from ..model import Task
from .base import Environment, StepResult

_FAMILIES = {
    "Heat and place": ("heat an object and place it in a receptacle", "put a hot {o} in {r}", "heat"),
    "Cool and place": ("cool an object and place it in a receptacle", "put a cool {o} in {r}", "cool"),
    "Clean and place": ("clean an object and place it in a receptacle", "put a clean {o} in {r}", "clean"),
}
_OBJECTS = ["cup", "mug", "apple", "egg", "plate", "bowl"]
_RECEPTACLES = ["cabinet", "fridge", "countertop", "shelf"]


class ToyEnv(Environment):
    name = "toy"
    context = ("A kitchen. Each turn, reply with exactly one of the admissible actions listed in the "
               "observation. Actions: take <object>, heat|cool|clean <object>, put <object> in <receptacle>.")

    def __init__(self, max_steps: int = 6, **_):
        self.max_steps = max_steps

    def tasks(self, split: str = "eval") -> list[Task]:
        out, shift = [], 0 if split == "train" else 1
        for family, (template, desc, _) in _FAMILIES.items():
            for i, o in enumerate(_OBJECTS):
                r = _RECEPTACLES[(i + shift) % len(_RECEPTACLES)]
                out.append(Task(f"{split}-{family.split()[0].lower()}-{o}-{r}", family, template,
                                desc.format(o=o, r=r)))
        return out

    def reset(self, task: Task) -> str:
        verb = _FAMILIES[task.task_family][2]
        obj, recep = task.task_id.split("-")[2:4]
        self.plan = [f"take {obj}", f"{verb} {obj}", f"put {obj} in {recep}"]
        distractor = next(o for o in _OBJECTS if o != obj)
        self.admissible = self.plan + [f"take {distractor}", f"{verb} {distractor}", "look"]
        self.done_subtasks = 0
        return self._obs(f"You are in a kitchen. Your task is to: {task.task_desc}.")

    def _obs(self, text: str) -> str:
        return f"{text}\nAdmissible actions: {'; '.join(sorted(self.admissible))}"

    def step(self, action: str) -> StepResult:
        if action.strip() == self.plan[self.done_subtasks]:
            self.done_subtasks += 1
            if self.done_subtasks == len(self.plan):
                return StepResult("Done.", 1.0, True, True)
            return StepResult(self._obs("OK."), 0.0, False)
        return StepResult(self._obs("Nothing happens."), 0.0, False)

    def progress(self) -> float:
        return self.done_subtasks / len(self.plan)
