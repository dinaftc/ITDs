"""Algorithm 1 on the toy environment with scripted solvers."""
import pytest

from itd.embeddings import HashEmbedder
from itd.envs.toy import ToyEnv
from itd.lifecycle import Lifecycle
from itd.model import FAILURE, NEAR_SUCCESS, SUCCESS
from itd.solver import TaskSolver, parse_action
from itd.store import make_store


class Scripted:
    """Plays the plan of the toy task up to `stop` subtasks; records its prompts."""
    name = "scripted"
    seconds = 0.0

    def __init__(self, stops):
        self.stops, self.attempt, self.systems, self.synth_calls = list(stops), -1, [], 0

    def chat(self, messages):
        if messages[0]["role"] != "system":          # synthesis request
            self.synth_calls += 1
            return "<think>hidden</think>DO: take the requested object first."
        if len(messages) == 2:                       # first turn of a new attempt
            self.attempt += 1
            self.systems.append(messages[0]["content"])
        done = (len(messages) - 2) // 2
        task = messages[1]["content"].split("Your task is to: ")[1].split(".")[0]
        verb = {"hot": "heat", "cool": "cool", "clean": "clean"}[task.split()[2]]
        obj, recep = task.split()[3], task.split()[5]
        plan = [f"take {obj}", f"{verb} {obj}", f"put {obj} in {recep}"]
        return f"Thought: next.\nAction: {plan[done] if done < self.stops[self.attempt] else 'look'}"


def make(stops, **kw):
    env, llm = ToyEnv(), Scripted(stops)
    lc = Lifecycle(make_store("memory", HashEmbedder()), TaskSolver(llm, env), llm, **kw)
    return lc, llm, env.tasks()


def test_parse_action():
    assert parse_action("Thought: x\nAction: take cup") == "take cup"
    assert parse_action("<think>Action: no</think>Action: `click('12')`\n") == "click('12')"
    assert parse_action("take cup") == "take cup"


def test_online_on_demand_populates_and_reuses():
    lc, llm, tasks = make([3, 2, 0, 3], approach="online", policy="on-demand", strategy="rules")
    heat = [t for t in tasks if t.task_family == "Heat and place"]
    labels = [lc.run(t)[0].quality_category for t in heat[:3]]
    assert labels == [SUCCESS, NEAR_SUCCESS, FAILURE]
    assert lc.store.count_traces() == 3
    assert "BEHAVIORAL RULES" not in llm.systems[0]          # empty ITD: nothing to reuse
    assert "DO: take the requested object first." in llm.systems[1] and "hidden" not in llm.systems[1]
    trace, info = lc.run(heat[3])
    assert info["experience_traces"] == 3 and info["attempts"] == 1
    # every Maintain call changed the template's partitions, so each task re-synthesized
    assert llm.synth_calls == 3


def test_offline_does_not_update_and_caches_experience():
    lc, llm, tasks = make([3, 3, 3], approach="online", strategy="rules")
    lc.run(tasks[0])
    lc.approach = "offline"
    lc.run(tasks[1]), lc.run(tasks[2])
    assert lc.store.count_traces() == 1
    assert llm.synth_calls == 1                               # second task hits the cache


def test_failure_triggered_keeps_failed_first_attempt_transient():
    lc, llm, tasks = make([3, 0, 2, 3], approach="online", policy="failure-triggered", strategy="examples")
    trace, info = lc.run(tasks[0])                            # unaided success: stored, no retrieval
    assert (info["attempts"], info["first_attempt_success"]) == (1, True)
    trace, info = lc.run(tasks[1])                            # fails unaided, retries with experience
    assert (info["attempts"], info["first_attempt_success"]) == (2, False)
    assert "PAST EXPERIENCES" not in llm.systems[1] and "PAST EXPERIENCES" in llm.systems[2]
    assert info["experience_traces"] == 1                     # only the trace retained before this task
    assert trace.quality_category == NEAR_SUCCESS and lc.store.count_traces() == 2


def test_combined_and_leave_one_out():
    lc, llm, tasks = make([3, 3, 3], approach="online", strategy="combined", leave_one_out=True)
    lc.run(tasks[0])
    lc.approach = "offline"
    _, info = lc.run(tasks[0])                                # its own trace is excluded
    assert info["experience_traces"] == 0
    lc.run(tasks[1])
    for marker in ("PAST EXPERIENCES", "BEHAVIORAL RULES", "REFLECTION FROM"):
        assert marker in llm.systems[2]


def test_offline_requires_no_writes_under_failure_trigger():
    lc, _, tasks = make([3], approach="offline", policy="failure-triggered")
    lc.run(tasks[0])
    assert lc.store.count_traces() == 0
