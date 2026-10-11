"""ALFWorld adapter (text-only AlfredTWEnv).

Family is the PDDL ``task_type``, template the goal sentence with its targets
slotted out; both derived from ``traj_data.json``. Progress compares the
planner's plan at reset with the plan remaining after execution, excluding
navigation.

Requires `alfworld-download` and ALFWORLD_DATA.
"""
from __future__ import annotations

import glob
import json
import os
import random
import re

import yaml

from ..model import Task
from .base import Environment, StepResult

# The six PDDL task types. The family is the task_type itself; the template is
# the goal sentence with its targets slotted out.
TEMPLATES = {
    "pick_and_place_simple": "put a {object} in {receptacle}",
    "pick_two_obj_and_place": "put two {object} in {receptacle}",
    "look_at_obj_in_light": "look at {object} under the {toggle}",
    "pick_clean_then_place_in_recep": "put a clean {object} in {receptacle}",
    "pick_heat_then_place_in_recep": "put a hot {object} in {receptacle}",
    "pick_cool_then_place_in_recep": "put a cool {object} in {receptacle}",
}
SPLITS = {"eval": "valid_unseen", "valid_unseen": "valid_unseen", "valid_seen": "valid_seen", "train": "train"}

CONTEXT = """You are a household agent acting in a text-based simulated home (ALFWorld).
Available actions (one per turn):
go to <receptacle>; open <receptacle>; close <receptacle>; take <object> from <receptacle>;
move <object> to <receptacle>; heat <object> with <receptacle>; cool <object> with <receptacle>;
clean <object> with <receptacle>; use <object>; examine <object>; look; inventory.
Objects and receptacles are named with an index, e.g. "mug 1", "cabinet 3". If an action has no
effect, the observation is "Nothing happens."."""


def _is_navigation(command: str) -> bool:
    return command.startswith("go to")


class ALFWorldEnv(Environment):
    name = "alfworld"
    context = CONTEXT

    def __init__(self, config: str = "configs/alfworld_base.yaml", max_steps: int = 50,
                 train_tasks: int = 180, train_seed: int = 0, **_):
        """config: ALFWorld's base_config.yaml. train_tasks: size of the sample
        of training games (balanced over families) used to pre-populate a store."""
        self.max_steps, self.train_tasks, self.train_seed = max_steps, train_tasks, train_seed
        with open(config) as f:
            self.config = yaml.safe_load(f)
        self.config["env"]["type"] = "AlfredTWEnv"
        # The planner expert exposes the remaining plan through infos["extra.expert_plan"].
        self.config["env"]["expert_type"] = "planner"
        self.config["general"]["training_method"] = "dagger"
        self.config["dagger"]["training"]["max_nb_steps_per_episode"] = max_steps + 1
        self._games: dict[str, str] = {}
        self._env = self._tw = None

    def tasks(self, split: str = "eval") -> list[Task]:
        root = os.path.join(os.path.expandvars("$ALFWORLD_DATA"), "json_2.1.1", SPLITS[split])
        by_family: dict[str, list[Task]] = {}
        for game in sorted(glob.glob(os.path.join(root, "*", "*", "game.tw-pddl"))):
            with open(os.path.join(os.path.dirname(game), "traj_data.json")) as f:
                traj = json.load(f)
            if traj["task_type"] not in TEMPLATES:
                continue
            with open(game) as f:
                data = json.load(f)
            if not data.get("solvable", True):
                continue
            family = traj["task_type"]
            template = TEMPLATES[family]
            task_id = "/".join(game.split(os.sep)[-3:-1])
            self._games[task_id] = game
            by_family.setdefault(family, []).append(Task(task_id, family, template, self._goal(data, traj)))
        if split == "train" and self.train_tasks:
            rng, per_family = random.Random(self.train_seed), self.train_tasks // len(by_family)
            by_family = {f: rng.sample(ts, min(per_family, len(ts))) for f, ts in by_family.items()}
        return [t for f in sorted(by_family) for t in by_family[f]]

    @staticmethod
    def _goal(game: dict, traj: dict) -> str:
        """The goal sentence of the game, e.g. "put a hot cup in cabinet"."""
        m = re.search(r'Your task is to: ([^.\n"\\]+)', game.get("grammar", ""))
        if m and "UNKNOWN" not in m.group(1):
            return m.group(1).strip()
        p = {k: (v or "").lower() for k, v in traj["pddl_params"].items() if isinstance(v, str)}
        obj, recep = p.get("object_target", ""), p.get("parent_target", "")
        return {
            "pick_and_place_simple": f"put a {obj} in {recep}",
            "pick_two_obj_and_place": f"put two {obj} in {recep}",
            "look_at_obj_in_light": f"look at {obj} under the {p.get('toggle_target', '')}",
            "pick_clean_then_place_in_recep": f"put a clean {obj} in {recep}",
            "pick_heat_then_place_in_recep": f"put a hot {obj} in {recep}",
            "pick_cool_then_place_in_recep": f"put a cool {obj} in {recep}",
        }[traj["task_type"]]

    def reset(self, task: Task) -> str:
        from alfworld.agents.environment import get_environment

        if self._env is not None:
            self._env.close()
        if self._tw is None:    # built once: the constructor scans a whole split for game files
            self._tw = get_environment("AlfredTWEnv")(self.config, train_eval="eval_out_of_distribution")
        self._tw.game_files, self._tw.num_games = [self._games[task.task_id]], 1
        self._env = self._tw.init_env(batch_size=1)
        obs, infos = self._env.reset()
        self._initial = self._remaining = self._plan(infos)
        # drop the "-= Welcome to TextWorld, ALFRED! =-" banner
        return "\n".join(obs[0].split("\n\n")[1:]).strip()

    @staticmethod
    def _plan(infos) -> int | None:
        plan = infos.get("extra.expert_plan")
        if not plan or plan[0] is None:
            return None
        return sum(not _is_navigation(c) for c in plan[0])

    def step(self, action: str) -> StepResult:
        obs, _, dones, infos = self._env.step([action])
        won = bool(infos["won"][0])
        remaining = self._plan(infos)
        if remaining is not None:
            self._remaining = remaining
        return StepResult(obs[0].strip(), float(won), bool(dones[0]) or won, won)

    def progress(self) -> float:
        if not self._initial or self._remaining is None:
            return 0.0
        return min(max((self._initial - self._remaining) / self._initial, 0.0), 1.0)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
