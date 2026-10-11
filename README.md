# Interaction Trace Databases

An LLM agent that solves a task produces a trace: what it saw, what it did,
and whether it worked. An ITD stores those traces, bounds how many it keeps as
more arrive, and returns the relevant ones when a similar task comes up. What
is made of them afterwards is a separate decision, so the same retained traces
can be replayed as demonstrations, distilled into rules, or turned into a
retrospective.

Two operators:

- **`maintain(trace, k)`** — keep at most `k` traces per partition, a
  partition being one (task template, quality category). The three categories
  are bounded separately, so one cannot displace another, and the store holds
  at most `3k|T|` traces for `|T|` templates.
- **`retrieve(task, category, n)`** — up to `n` traces of one quality
  category. Traces of the task's own template come first, which is an indexed
  equality lookup; if that does not fill `n`, the rest come from other
  templates of the same family, ranked by embedding similarity.
  `retrieve_all` calls it once per category, returning at most `3n`.

```python
from itd.embeddings import SentenceEmbedder
from itd.model import Task, Trace, Step
from itd.quality import assign_quality
from itd.store import make_store

store = make_store("postgres", SentenceEmbedder(), "demo",
                   dsn="postgresql://itd:itd@localhost/itd")

task = Task("308", "gitlab",
            "Tell me who has made the most contributions to {repo}",
            "Tell me who has made the most contributions, in terms of number "
            "of commits, to the a11yproject.com project")
trace = Trace(task.task_id, task.task_family, task.template_desc,
              actor="deepseek-v3.2", label="success", reward=1.0, horizon=4,
              steps=[Step(0, "<accessibility tree>", "click('1194')")])

store.add_task(task)
store.maintain(assign_quality(trace), k=1)
store.retrieve_all(task, n=1)
```

[`examples/`](examples/) holds real traces and experiences from actual runs.

## Layout

| Path | Content |
|---|---|
| `itd/model.py` | `Task`, `Trace`, `Step`, `Experience` |
| `itd/quality.py` | quality categories, scores, the near-success threshold θ |
| `itd/store/base.py` | the operator interface and the experience cache |
| `itd/store/postgres.py`, `schema.sql` | PostgreSQL: B-tree on the partition key and ranking, pgvector HNSW over template embeddings |
| `itd/store/qdrant.py` | Qdrant: trace and template collections, payload indices, HNSW |
| `itd/store/memory.py` | in-memory backend, the reference the other two are tested against |
| `itd/strategies.py`, `prompts.py` | Examples, Rules, Reflections, Combined; synthesis prompts and injection wrappers |
| `itd/lifecycle.py` | retrieve, solve, maintain, for one task |
| `itd/solver.py` | zero-shot ReAct task solver |
| `itd/envs/` | ALFWorld and WebArena adapters |
| `itd/runner.py`, `cli.py` | running configurations, raw and Retry baselines, summaries |
| `itd/bench.py` | Maintain and Retrieve latency on a synthetic store |
| `scripts/experiments.sh` | every experiment, one function each |
| `tests/` | operator semantics on all three backends |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[postgres,qdrant,embeddings,dev]"
docker compose up -d            # PostgreSQL 16 + pgvector 0.8.0, Qdrant 1.19.2
cp .env.example .env            # endpoints and API keys
pytest                          # set ITD_TEST_PG_DSN to include PostgreSQL
```

Task solvers and synthesizers are declared in `configs/models.yaml` and
reached over OpenAI-compatible endpoints; check the `model` identifiers
against your deployment.

**ALFWorld** needs `alfworld-download`, `ALFWORLD_DATA` set, and ALFWorld's
`configs/base_config.yaml` copied to `configs/alfworld_base.yaml`.

**WebArena** needs `playwright install chromium`, the `WA_*` variables of a
running instance, and the intermediate checkpoints in
`data/webarena_checkpoints.json` (format in
`data/webarena_checkpoints.example.json`). The checkpoints only rank
unsuccessful traces; the benchmark's own evaluation decides success and is
never replaced.

## Running

`itd run` runs one configuration. A YAML file supplies the defaults
(`configs/alfworld.yaml`, `configs/webarena.yaml`) and flags override them.
Each run writes `config.json`, `results.jsonl` (one record per task, with
timings) and `traces.jsonl` to `runs/<benchmark>/<name>_seed<seed>/`, against
its own store.

```bash
itd run --config configs/alfworld.yaml --mode raw --split train --tag populate
itd run --config configs/alfworld.yaml --approach offline --policy on-demand \
        --strategy rules --init-traces runs/alfworld/populate_seed0/traces.jsonl
itd summarize runs/alfworld/*
```

Three choices, varied independently:

- **Maintenance approach** — *Offline* freezes the store before evaluation,
  *Online* starts empty and grows it as tasks are completed, *Hybrid* starts
  pre-populated and keeps growing.
- **Retrieval policy** — *On-demand* retrieves before the first attempt.
  *Failure-triggered* attempts the task unaided first and retrieves only after
  a failure; that failed attempt is never written back, so the retry sees only
  what was retained before the task. It spends a second attempt, so the
  comparison is `--mode retry`, a second unaided attempt.
- **Reuse strategy** — Examples, Rules, Reflections, or Combined.

Offline and Hybrid need a seed store, loaded from the `traces.jsonl` of a run
that reused nothing. It is loaded through `maintain`, so `k` and θ shape it as
they shape a growing store.

`scripts/experiments.sh` runs each experiment: `populate_*`, `sweep_*`,
`transfer_*`, `ablation_categories`, `ablation_budgets`, `ablation_theta`, and
`bench`.

## Quality

| Category | When | Score |
|---|---|---|
| success | the benchmark says the task was solved | `reward / horizon` — shorter is better |
| near-success | not solved, progress `m/M ≥ θ` | `m/M` — further is better |
| failure | not solved, progress below θ | `1 - m/M` — failing early is better |

Progress comes from the environment, never from the agent's own text: ALFWorld
compares the planner's plan at reset with the plan remaining after execution,
excluding navigation; WebArena matches intermediate checkpoints. Retrieval
reads all three categories by default.
