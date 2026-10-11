"""Experiment runner: one run = one configuration over a list of tasks."""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .embeddings import make_embedder
from .envs.base import make_env
from .lifecycle import HYBRID, OFFLINE, Lifecycle
from .llm import make_llm
from .model import CATEGORIES, SUCCESS, Task, Trace
from .quality import assign_quality
from .solver import TaskSolver
from .store import make_store


@dataclass
class RunConfig:
    benchmark: str = "alfworld"            # alfworld | webarena
    split: str = "eval"
    mode: str = "itd"                      # raw | retry | itd
    approach: str = "offline"              # offline | online | hybrid
    policy: str = "on-demand"              # on-demand | failure-triggered
    strategy: str = "rules"                # examples | rules | reflections | combined
    k: int = 1
    n: int = 1
    theta: float = 0.5
    categories: str = ",".join(CATEGORIES)  # quality categories available to RetrieveAll
    solver: str = "qwen2.5-72b"            # model names from configs/models.yaml
    synthesizer: str = ""                  # defaults to the solver
    init_traces: str = ""                  # JSONL of traces pre-populating Offline / Hybrid ITDs
    leave_one_out: bool = False
    families: str = ""                     # comma-separated task families ("" = all)
    limit: int = 0                         # evaluate only the first N tasks (0 = all)
    shuffle: bool = False                  # randomize task arrival order with `seed`
    seed: int = 0
    store: str = "memory"                  # memory | postgres | qdrant
    dsn: str = ""
    qdrant_url: str = ""
    embedder: str = "all-MiniLM-L6-v2"     # or "hash"
    history_obs: int = 0                   # recent observations kept in the prompt (0 = all)
    max_obs_chars: int = 300               # observation budget per step when rendering traces
    env: dict | None = None                # benchmark-specific options
    out: str = "runs"
    tag: str = ""                          # run name; derived from the configuration if empty

    def name(self) -> str:
        if self.tag:
            return self.tag
        parts = [self.benchmark, self.solver, self.mode]
        if self.mode == "itd":
            parts += [self.approach, self.policy, self.strategy]
        return "_".join(parts)


def namespace(cfg: RunConfig) -> str:
    """Schema / collection prefix of the run's ITD (short enough for PostgreSQL identifiers)."""
    return "itd_" + hashlib.md5(f"{cfg.name()}_seed{cfg.seed}".encode()).hexdigest()[:12]


def load_traces(path: str) -> list[Trace]:
    with open(path) as f:
        return [Trace.from_dict(json.loads(line)) for line in f if line.strip()]


def populate(lc: Lifecycle, traces: list[Trace]) -> None:
    """Pre-populate an ITD by passing stored traces through Maintain."""
    for t in traces:
        t.quality_category = t.quality_score = None      # re-assigned with this run's theta
        lc.maintain(Task(t.task_id, t.task_family, t.template_desc, t.task_desc or ""), t)


def run(cfg: RunConfig) -> dict:
    out = Path(cfg.out) / f"{cfg.name()}_seed{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    env = make_env(cfg.benchmark, **(cfg.env or {}))
    tasks = env.tasks(cfg.split)
    if cfg.families:
        keep = {f.strip().lower() for f in cfg.families.split(",")}
        tasks = [t for t in tasks if t.task_family.lower() in keep]
    if cfg.shuffle:
        random.Random(cfg.seed).shuffle(tasks)
    if cfg.limit:
        tasks = tasks[: cfg.limit]

    llm = make_llm(cfg.solver, seed=cfg.seed)
    synthesizer = make_llm(cfg.synthesizer, seed=cfg.seed) if cfg.synthesizer else llm
    solver = TaskSolver(llm, env, cfg.history_obs or None)
    lc = None
    if cfg.mode == "itd":
        store = make_store(cfg.store, make_embedder(cfg.embedder), namespace(cfg), reset=True, dsn=cfg.dsn or None, url=cfg.qdrant_url or None)
        lc = Lifecycle(store, solver, synthesizer, cfg.approach, cfg.policy, cfg.strategy, cfg.k, cfg.n,
                       cfg.theta, tuple(c.strip() for c in cfg.categories.split(",")), cfg.leave_one_out,
                       cfg.max_obs_chars)
        if cfg.approach in (OFFLINE, HYBRID):
            if not cfg.init_traces:
                raise ValueError(f"{cfg.approach} maintenance needs init_traces")
            populate(lc, load_traces(cfg.init_traces))
            print(f"[itd] pre-populated with {store.count_traces()} traces")

    with open(out / "results.jsonl", "w") as results, open(out / "traces.jsonl", "w") as traces:
        for i, task in enumerate(tasks, 1):
            start = time.perf_counter()
            llm0, env0, syn0 = llm.seconds, solver.env_seconds, synthesizer.seconds
            if lc:
                lc.timing = {"retrieve_s": 0.0, "synthesis_s": 0.0}
                trace, info = lc.run(task)
            else:
                trace, info = solver.complete(task), {"attempts": 1}
                if cfg.mode == "retry" and trace.label != SUCCESS:
                    trace, info = solver.complete(task), {"attempts": 2}
            if trace.quality_category is None:
                assign_quality(trace, cfg.theta)
            synthesis_s = lc.timing["synthesis_s"] if lc else 0.0
            record = {
                "task_id": task.task_id, "task_family": task.task_family, "template_desc": task.template_desc,
                "success": trace.label == SUCCESS, "reward": trace.reward, "steps": trace.horizon,
                "quality_category": trace.quality_category, **info, **env.task_meta(task),
                "total_s": time.perf_counter() - start, "env_s": solver.env_seconds - env0,
                "retrieve_s": lc.timing["retrieve_s"] if lc else 0.0, "synthesis_s": synthesis_s,
                # with synthesizer == solver, llm.seconds also counts synthesis calls
                "llm_s": llm.seconds - llm0 - (synthesizer.seconds - syn0 if synthesizer is llm else 0.0),
            }
            results.write(json.dumps(record) + "\n")
            results.flush()
            traces.write(json.dumps(trace.to_dict()) + "\n")
            traces.flush()
            print(f"[{i}/{len(tasks)}] {task.task_id}: {trace.label} in {trace.horizon} steps")
    env.close()
    if lc:
        lc.store.close()
    summary = summarize([out])[0]
    print(json.dumps(summary))
    return summary


def summarize(run_dirs: list[Path], by: str | None = None) -> list[dict]:
    """Success rate per configuration. A single run reports the binomial
    standard error over its tasks (WebArena protocol); several seeds of the
    same configuration report mean ± s.d. over runs (ALFWorld protocol)."""
    groups: dict[tuple, list[list[bool]]] = defaultdict(list)
    for d in sorted(Path(p) for p in run_dirs):
        if not (d / "results.jsonl").exists():
            continue
        cfg = RunConfig(**json.loads((d / "config.json").read_text()))
        rows = [json.loads(line) for line in (d / "results.jsonl").read_text().splitlines()]
        split: dict = defaultdict(list)
        for r in rows:
            split[str(r.get(by)) if by else "all"].append(r["success"])
        for value, successes in split.items():
            groups[(cfg.name(), value)].append(successes)
    out = []
    for (name, value), runs in groups.items():
        rates = [100 * sum(r) / len(r) for r in runs]
        row = {"config": name, "group": value, "runs": len(runs), "tasks": len(runs[0]),
               "success_rate": round(statistics.mean(rates), 1)}
        if len(runs) > 1:
            row["sd"] = round(statistics.stdev(rates), 1)
        else:
            p = rates[0] / 100
            row["binomial_se"] = round(100 * math.sqrt(p * (1 - p) / len(runs[0])), 1)
        out.append(row)
    return out
