"""Operator efficiency (Section 4.8): median Maintain / Retrieve latency on
synthetic ITDs, varying the number of traces, the number of task templates,
and the maintenance and retrieval budgets.

Data preparation is excluded from timing and query embeddings are precomputed.
"""
from __future__ import annotations

import csv
import math
import statistics
import time
import uuid
from dataclasses import dataclass

import numpy as np

from .model import CATEGORIES, SUCCESS, Step, Task, Trace
from .store import EXACT_FALLBACK, SIMILARITY_ONLY, make_store
from .store.memory import MemoryStore

FAMILIES = 6
DIM = 384
_WARMUP_MAINTAIN = 5


class _NoEmbedder:
    """Embeddings are synthetic and precomputed; the stores must not embed."""
    dim = DIM

    def encode(self, texts):
        raise RuntimeError("bench passes precomputed embeddings")


def add_arguments(p) -> None:
    p.add_argument("--backends", default="postgres,qdrant", help="comma-separated: memory, postgres, qdrant")
    p.add_argument("--experiments", default="traces,templates,budgets")
    p.add_argument("--trace-counts", default="1000,10000,100000,1000000")
    p.add_argument("--template-counts", default="190,1000,10000,100000,1000000")
    p.add_argument("--budgets", default="1,3,5,10,50")
    p.add_argument("--templates", type=int, default=190)
    p.add_argument("--budget-traces", type=int, default=100_000)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--retrieve-calls", type=int, default=200)
    p.add_argument("--maintain-calls", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--fallback-frac", type=float, default=0.4,
                   help="share of requests on unseen templates in the template-count experiment")
    p.add_argument("--steps", type=int, default=3, help="steps per synthetic trace")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dsn", default=None)
    p.add_argument("--qdrant-url", default=None)
    p.add_argument("--out", default="runs/bench.csv")


# -- synthetic ITDs ----------------------------------------------------------

def _unit(rng, rows: int) -> np.ndarray:
    v = rng.standard_normal((rows, DIM)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _task(i: int) -> Task:
    return Task(f"task-{i}", f"family-{i % FAMILIES}", f"template {i}", f"task of template {i}")


def _trace(task: Task, category: str, rng, steps: int, score: float | None = None) -> Trace:
    horizon = int(rng.integers(1, 31))
    return Trace(task.task_id, task.task_family, task.template_desc, "synthetic",
                 "success" if category == SUCCESS else "failure", float(category == SUCCESS), horizon,
                 [Step(s, f"observation {s}", f"action {s}") for s in range(steps)],
                 trace_id=str(uuid.UUID(bytes=rng.bytes(16))), quality_category=category,
                 quality_score=float(rng.random()) * 0.9 if score is None else score, task_desc=task.task_desc)


@dataclass
class Dataset:
    tasks: list[Task]
    vecs: np.ndarray
    reserved: int             # templates [0, reserved) hold exactly k traces per partition (Maintain targets)
    per_partition: int        # traces in each partition of the other templates

    def traces(self, rng, k: int, steps: int):
        for i, task in enumerate(self.tasks):
            for c in CATEGORIES:
                for _ in range(k if i < self.reserved else self.per_partition):
                    yield _trace(task, c, rng, steps)


def _load(store, data: Dataset, rng, k: int, steps: int) -> None:
    traces = data.traces(rng, k, steps)
    if isinstance(store, MemoryStore):
        for task, v in zip(data.tasks, data.vecs):
            store.tasks[task.task_id] = task
            store.templates[task.task_family][task.template_desc] = v
        for t in traces:
            store.partitions[(t.task_family, t.template_desc, t.quality_category)].append(t)
        for part in store.partitions.values():
            part.sort(key=lambda t: (-t.quality_score, t.horizon))
    elif type(store).__name__ == "PostgresStore":
        cur = store.conn.cursor()
        cur.execute("DROP INDEX template_hnsw_idx")     # rebuilt after the bulk load
        with cur.copy("COPY task FROM STDIN") as cp:
            for t in data.tasks:
                cp.write_row((t.task_id, t.task_family, t.template_desc, t.task_desc))
        with cur.copy("COPY template FROM STDIN") as cp:
            for t, v in zip(data.tasks, data.vecs):
                cp.write_row((t.task_family, t.template_desc, "[" + ",".join(map(str, v.tolist())) + "]"))
        batch: list[Trace] = []

        def flush():
            with cur.copy("COPY trace (trace_id, task_id, task_family, template_desc, actor, label, reward, "
                          "horizon, quality_category, quality_score) FROM STDIN") as cp:
                for t in batch:
                    cp.write_row((t.trace_id, t.task_id, t.task_family, t.template_desc, t.actor, t.label,
                                  t.reward, t.horizon, t.quality_category, t.quality_score))
            with cur.copy("COPY step FROM STDIN") as cp:
                for t in batch:
                    for s in t.steps:
                        cp.write_row((t.trace_id, s.step_index, s.observation, s.action))
            batch.clear()

        for t in traces:
            batch.append(t)
            if len(batch) >= 50_000:
                flush()
        flush()
        cur.execute("CREATE INDEX template_hnsw_idx ON template USING hnsw (embedding vector_cosine_ops)")
        cur.execute("ANALYZE")
    else:   # Qdrant
        from qdrant_client import models as qm

        from .store.qdrant import _tid
        store.client.upload_points(store.templates, (
            qm.PointStruct(id=_tid(t.task_family, t.template_desc), vector=v.tolist(),
                           payload={"task_family": t.task_family, "template_desc": t.template_desc})
            for t, v in zip(data.tasks, data.vecs)), batch_size=512, wait=True)
        store.client.upload_points(store.traces, (store.trace_point(t) for t in traces), batch_size=512, wait=True)


def _noop(store) -> float | None:
    """Median cost of a request without data access."""
    if type(store).__name__ == "PostgresStore":
        fn = lambda: store.conn.execute("SELECT 1").fetchone()
    elif type(store).__name__ == "QdrantStore":
        fn = lambda: store.client.collection_exists(store.traces)
    else:
        return None
    return statistics.median(_time(fn) for _ in range(200))


def _time(fn) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1e3


# -- measurements ------------------------------------------------------------

def measure(args, backend: str, n_templates: int, n_traces: int | None, k: int, n: int,
            fallback_frac: float, rng) -> list[dict]:
    """Load one synthetic ITD into `backend` and time both operators on it."""
    reserved = math.ceil((args.maintain_calls + _WARMUP_MAINTAIN) / len(CATEGORIES)) if n_traces else 0
    if n_traces:      # trace-count / budget experiments: N traces over a fixed set of templates
        if reserved >= n_templates:
            raise ValueError("not enough templates for the requested number of Maintain calls")
        rest = n_traces - reserved * len(CATEGORIES) * k
        per_partition = max(1, round(rest / ((n_templates - reserved) * len(CATEGORIES))))
    else:             # template-count experiment: one trace per quality category and template
        per_partition = 1
    data = Dataset([_task(i) for i in range(n_templates)], _unit(rng, n_templates), reserved, per_partition)
    store = make_store(backend, _NoEmbedder(), "itd_bench", reset=True, dsn=args.dsn, url=args.qdrant_url)
    _load(store, data, rng, k, args.steps)

    # Retrieve requests: a known template (exact lookup) or an unseen one (similarity fallback).
    total = args.warmup + args.retrieve_calls
    known = rng.integers(reserved, n_templates, size=total)
    unseen = rng.random(total) < fallback_frac
    queries = _unit(rng, total)
    requests = []
    for j in range(total):
        if unseen[j]:
            fam = f"family-{j % FAMILIES}"
            requests.append((Task(f"new-{j}", fam, f"unseen template {j}", "new task"), queries[j]))
        else:
            requests.append((data.tasks[known[j]], data.vecs[known[j]]))

    rows = []
    for mode in (EXACT_FALLBACK, SIMILARITY_ONLY):
        times = [_time(lambda: store.retrieve_all(task, n, query_vec=q, mode=mode)) for task, q in requests]
        times = times[args.warmup:]
        rows.append({"operator": "retrieve", "mode": mode, "median_ms": statistics.median(times),
                     "mean_ms": statistics.mean(times)})
        if mode == EXACT_FALLBACK and 0 < fallback_frac < 1:
            for name, flag in (("exact", False), ("fallback", True)):
                part = [t for t, u in zip(times, unseen[args.warmup:]) if u == flag]
                rows.append({"operator": "retrieve", "mode": f"{mode}/{name}",
                             "median_ms": statistics.median(part), "mean_ms": statistics.mean(part)})

    # Maintain: each call targets a distinct partition; with `reserved` partitions it
    # holds exactly k traces and the incoming trace outranks them, so the call writes.
    total = _WARMUP_MAINTAIN + args.maintain_calls
    targets = [(i // len(CATEGORIES), CATEGORIES[i % len(CATEGORIES)]) for i in range(total)] if reserved else \
        [(int(i), CATEGORIES[j % len(CATEGORIES)]) for j, i in enumerate(rng.integers(0, n_templates, size=total))]
    incoming = [_trace(data.tasks[i], c, rng, args.steps, score=0.95) for i, c in targets]
    times = [_time(lambda: store.maintain(t, k)) for t in incoming][_WARMUP_MAINTAIN:]
    rows.append({"operator": "maintain", "mode": "", "median_ms": statistics.median(times),
                 "mean_ms": statistics.mean(times)})
    noop = _noop(store)
    if noop is not None:
        rows.append({"operator": "noop", "mode": "", "median_ms": noop, "mean_ms": noop})
    store.close()
    return rows


def main(args) -> None:
    ints = lambda s: [int(x) for x in s.split(",") if x]
    plan = []       # (experiment, x, templates, traces, k, n, fallback_frac)
    for exp in args.experiments.split(","):
        if exp == "traces":
            plan += [(exp, x, args.templates, x, args.k, args.n, 0.0) for x in ints(args.trace_counts)]
        elif exp == "templates":
            plan += [(exp, x, x, None, args.k, args.n, args.fallback_frac) for x in ints(args.template_counts)]
        elif exp == "budgets":
            plan += [("budget_n", x, args.templates, args.budget_traces, args.k, x, 0.0) for x in ints(args.budgets)]
            plan += [("budget_k", x, args.templates, args.budget_traces, x, args.n, 0.0) for x in ints(args.budgets)]
        else:
            raise ValueError(f"unknown experiment: {exp}")
    fields = ["experiment", "backend", "x", "operator", "mode", "median_ms", "mean_ms"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for backend in args.backends.split(","):
            for exp, x, templates, traces, k, n, frac in plan:
                rng = np.random.default_rng(args.seed)
                for row in measure(args, backend, templates, traces, k, n, frac, rng):
                    # budget_n only varies Retrieve, budget_k only varies Maintain
                    if (exp, row["operator"]) in (("budget_n", "maintain"), ("budget_k", "retrieve")):
                        continue
                    row = {"experiment": exp, "backend": backend, "x": x, **row}
                    row["median_ms"], row["mean_ms"] = round(row["median_ms"], 3), round(row["mean_ms"], 3)
                    writer.writerow(row)
                    f.flush()
                    print(row)
