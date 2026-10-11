# Interaction Trace Databases

An LLM agent that solves a task produces a trace: what it saw, what it did,
and whether it worked. An ITD is a database for those traces. It keeps them,
decides which are worth keeping as more arrive, and hands the relevant ones
back when a similar task comes up — separately from how an agent chooses to
use them.

Two operators do the work:

- **`maintain(trace, k)`** — keep at most `k` traces per partition, where a
  partition is one (task template, quality category). The incoming trace
  competes with what is already there and the best `k` survive.
- **`retrieve(task, category, n)`** — up to `n` traces of one quality
  category. Traces of the task's own template come first; anything left over
  is filled from nearby templates of the same family, by embedding similarity.

Everything else is built on those two: three ways of turning retrieved traces
into something an agent can use, two points in a task's life at which to
retrieve, and three ways of deciding what goes into the store in the first
place.

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

See [`examples/`](examples/) for real traces and experiences, taken from
actual runs rather than written by hand.

## What is in the box

| Path | Content |
|---|---|
| `itd/model.py` | `Task`, `Trace`, `Step`, `Experience` |
| `itd/quality.py` | quality categories, scores, the near-success threshold θ |
| `itd/store/base.py` | the operator interface, plus the experience cache |
| `itd/store/postgres.py`, `schema.sql` | PostgreSQL backend: B-tree partitions, HNSW over template embeddings |
| `itd/store/qdrant.py` | Qdrant backend: trace and template collections |
| `itd/store/memory.py` | in-memory backend, the reference the other two are tested against |
| `itd/strategies.py`, `prompts.py` | Examples, Rules, Reflections, Combined; synthesis prompts and injection |
| `itd/lifecycle.py` | what happens to one task: retrieve, solve, maintain |
| `itd/solver.py` | a zero-shot ReAct solver |
| `itd/envs/` | ALFWorld and WebArena adapters, and a toy environment |
| `itd/runner.py`, `cli.py` | running configurations, baselines, summaries |
| `itd/bench.py` | operator latency against a synthetic store |
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
against your own deployment.

Both benchmarks are optional:

- **ALFWorld** — `pip install -e ".[alfworld]"`, `alfworld-download`, set
  `ALFWORLD_DATA`, and copy ALFWorld's `configs/base_config.yaml` to
  `configs/alfworld_base.yaml`.
- **WebArena** — `pip install -e ".[webarena]"`, `playwright install
  chromium`, and the `WA_*` variables of a running instance. Ranking
  unsuccessful traces needs intermediate checkpoints in
  `data/webarena_checkpoints.json`; the format is in
  `data/webarena_checkpoints.example.json`.

## Try it without any of that

A toy environment, a mock LLM that picks random admissible actions, and the
in-memory backend. No API key, no database, no benchmark:

```bash
itd run --config configs/toy.yaml --mode raw --split train --tag populate
itd run --config configs/toy.yaml --approach hybrid --policy failure-triggered \
        --strategy combined --init-traces runs/toy/populate_seed0/traces.jsonl
itd summarize runs/toy/*
```

It exercises the whole pipeline. The success rates it prints mean nothing.

## Running something real

`itd run` runs one configuration. A YAML file supplies the defaults
(`configs/alfworld.yaml`, `configs/webarena.yaml`) and flags override them;
`itd run --help` lists them. Each run writes `config.json`, `results.jsonl`
(one record per task, with timings) and `traces.jsonl` into
`runs/<benchmark>/<name>_seed<seed>/`, against its own store.

The three choices that matter:

- **When the store is written.** *Offline* freezes it before evaluation,
  *Online* grows it as tasks are solved, *Hybrid* starts from a seed store and
  keeps growing.
- **When it is read.** *On-demand* retrieves before the first attempt.
  *Failure-triggered* lets the agent try unaided and only retrieves if that
  fails — which costs a second attempt, so compare it against `--mode retry`,
  a second unaided attempt, and not against the single-attempt baseline.
- **What is handed over.** Examples, Rules, Reflections, or all three
  combined.

Offline and Hybrid need a seed store, loaded from a JSONL file of traces —
just the `traces.jsonl` of a run that reused nothing. It is loaded through
`maintain`, so `k` and θ shape the seed store exactly as they shape a growing
one.

## Quality

A trace's category decides which partition it competes in, and its score
decides whether it wins:

| Category | When | Score |
|---|---|---|
| success | the benchmark says the task was solved | `reward / horizon` — shorter is better |
| near-success | not solved, but progress `m/M ≥ θ` | `m/M` — further is better |
| failure | not solved, progress below θ | `1 - m/M` — failing early and cleanly is better |

Progress is whatever the environment can be asked for, never inferred from the
agent's own text: ALFWorld re-derives the remaining plan from world state,
WebArena matches intermediate checkpoints. Neither ever overrides the
benchmark's own verdict on success — they only rank what failed.

Retrieval reads all three categories by default. A failure is worth showing
precisely because it shows what not to do, which the Rules and Reflections
prompts both lean on.
