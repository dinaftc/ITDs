"""Task solver: a zero-shot ReAct loop (CompleteTask in Algorithm 1)."""
from __future__ import annotations

import re
import time

from .envs.base import Environment
from .model import FAILURE, SUCCESS, Step, Task, Trace

SYSTEM = """{context}

At every turn you receive an observation. Reply with a short reasoning, then exactly one action:
Thought: <reasoning>
Action: <action>"""


def parse_action(reply: str) -> str:
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
    m = re.findall(r"Action:\s*(.+?)(?=\n\s*(?:Thought|Observation):|\Z)", reply, flags=re.S)
    action = m[-1] if m else (reply.strip().splitlines() or [""])[-1]
    return action.strip().strip("`").strip()


class TaskSolver:
    def __init__(self, llm, env: Environment, history_obs: int | None = None):
        """history_obs: how many of the most recent observations stay in the
        prompt in full (None keeps all; WebArena pages are too long for that)."""
        self.llm, self.env, self.history_obs = llm, env, history_obs
        self.env_seconds = 0.0

    def _env(self, fn, *args):
        start = time.perf_counter()
        try:
            return fn(*args)
        finally:
            self.env_seconds += time.perf_counter() - start

    def complete(self, task: Task, experience: str | None = None) -> Trace:
        """One attempt at `task`, with `experience` (already wrapped) or without."""
        system = SYSTEM.format(context=self.env.context)
        if experience:
            system += "\n\n" + experience
        obs = self._env(self.env.reset, task)
        turns: list[tuple[str, str]] = []        # (observation, reply)
        steps: list[Step] = []
        result = None
        while len(steps) < self.env.max_steps:
            messages = [{"role": "system", "content": system}]
            for i, (o, reply) in enumerate(turns + [(obs, None)]):
                recent = self.history_obs is None or len(turns) - i < self.history_obs
                content = f"Observation: {o if recent else '[omitted]'}"
                if i == 0:
                    content = f"Task: {task.task_desc}\n\n{content}"
                messages.append({"role": "user", "content": content})
                if reply is not None:
                    messages.append({"role": "assistant", "content": reply})
            reply = self.llm.chat(messages)
            action = parse_action(reply)
            steps.append(Step(len(steps), obs, action))
            turns.append((obs, reply))
            result = self._env(self.env.step, action)
            obs = result.observation
            if result.done:
                break
        success = bool(result and result.success)
        return Trace(task_id=task.task_id, task_family=task.task_family, template_desc=task.template_desc,
                     actor=self.llm.name, label=SUCCESS if success else FAILURE,
                     reward=result.reward if result else 0.0, horizon=len(steps), steps=steps,
                     progress=1.0 if success else self._env(self.env.progress), task_desc=task.task_desc)
