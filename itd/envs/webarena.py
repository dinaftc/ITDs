"""WebArena adapter, through BrowserGym.

Task families come from the benchmark's `sites` field and task templates from
`intent_template` (190 templates over the 812 tasks). Progress m/M is computed
from intermediate checkpoints (see data/webarena_checkpoints.example.json):
they supplement the benchmark's final evaluation, are only used to assign
trace quality, and are never shown to the task solver.

Requires `pip install browsergym-webarena`, `playwright install chromium`,
and the WA_* environment variables of a running WebArena instance.
"""
from __future__ import annotations

import importlib.resources
import json
import re
from pathlib import Path

from ..model import Task
from .base import Environment, StepResult

SITE_FAMILY = {"shopping": "Shopping", "shopping_admin": "Shopping Admin", "reddit": "Reddit",
               "gitlab": "GitLab", "map": "Map"}

CONTEXT = """You are a web agent operating a browser to complete a task on a website.
Each observation gives the current URL and the accessibility tree of the page; elements are
identified by a [bid]. If your previous action failed, the error is reported.
Available actions (one per turn):
{actions}
When the task asks for information, answer with send_msg_to_user("<answer>"); if the task
cannot be completed, use report_infeasible("<reason>")."""


def load_task_configs() -> list[dict]:
    import webarena   # shipped with browsergym-webarena (libwebarena)

    return json.loads(importlib.resources.files(webarena).joinpath("test.raw.json").read_text())


def family_of(sites: list[str]) -> str:
    return SITE_FAMILY[sites[0]] if len(sites) == 1 else "Cross-site"


def interaction_type(config: dict) -> str:
    """Information-seeking tasks are evaluated on the returned answer only;
    state-changing tasks are evaluated on the resulting site state."""
    return "information-seeking" if config["eval"]["eval_types"] == ["string_match"] else "state-changing"


class WebArenaEnv(Environment):
    name = "webarena"

    def __init__(self, max_steps: int = 30, checkpoints: str = "data/webarena_checkpoints.json",
                 max_obs_chars: int = 40_000, headless: bool = True, **_):
        from browsergym.core.action.highlevel import HighLevelActionSet

        self.max_steps, self.max_obs_chars, self.headless = max_steps, max_obs_chars, headless
        self.action_set = HighLevelActionSet(subsets=["webarena"], strict=False, multiaction=False)
        self.context = CONTEXT.format(actions=self.action_set.describe(with_long_description=False,
                                                                       with_examples=True))
        self.configs = {str(c["task_id"]): c for c in load_task_configs()}
        path = Path(checkpoints)
        # {task_id: [checkpoint, ...]}; a checkpoint is {"url"|"action"|"obs": regex}
        self.checkpoints: dict[str, list[dict]] = json.loads(path.read_text()) if path.exists() else {}
        if not self.checkpoints:
            print(f"[webarena] no checkpoints at {path}: unsuccessful traces all get progress 0 (failure)")
        self._env = None

    def tasks(self, split: str = "eval") -> list[Task]:
        return [Task(tid, family_of(c["sites"]), c["intent_template"], c["intent"])
                for tid, c in self.configs.items()]

    def task_meta(self, task: Task) -> dict:
        return {"interaction_type": interaction_type(self.configs[task.task_id])}

    # -- episode -------------------------------------------------------------

    def reset(self, task: Task) -> str:
        import browsergym.webarena  # noqa: F401  (registers the gym environments)
        import gymnasium as gym

        self.close()
        self._env = gym.make(f"browsergym/webarena.{task.task_id}", headless=self.headless,
                             action_mapping=self.action_set.to_python_code)
        self._pending = list(self.checkpoints.get(task.task_id, []))
        self._total = len(self._pending)
        obs, _ = self._env.reset()
        return self._render(obs, action="")

    def step(self, action: str) -> StepResult:
        obs, reward, terminated, truncated, _ = self._env.step(action)
        return StepResult(self._render(obs, action), float(reward), terminated or truncated, reward > 0)

    def _render(self, obs: dict, action: str) -> str:
        from browsergym.utils.obs import flatten_axtree_to_str

        tree = flatten_axtree_to_str(obs["axtree_object"])[: self.max_obs_chars]
        seen = {"url": obs.get("url", ""), "action": action, "obs": tree}
        self._pending = [c for c in self._pending
                         if not all(re.search(pattern, seen[kind]) for kind, pattern in c.items())]
        text = f"URL: {obs.get('url', '')}\n{tree}"
        if obs.get("last_action_error"):
            text += f"\nError of the previous action: {obs['last_action_error']}"
        return text

    def progress(self) -> float:
        return (self._total - len(self._pending)) / self._total if self._total else 0.0

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
