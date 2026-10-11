# Interaction Trace Databases (ITDs)

An ITD stores the interaction traces produced by LLM agents and keeps their
storage, maintenance and retrieval separate from the way they are reused.
This repository contains the data model, the `Maintain` and `Retrieve`
operators over PostgreSQL/pgvector and Qdrant, the three reuse strategies
(Examples, Rules, Reflections, plus Combined), the On-demand and
Failure-triggered retrieval policies, and the scripts that run the paper's
experiments on WebArena and ALFWorld.

## Layout

| Path | Content |
|---|---|---|
| `itd/model.py` |  `Task`, `Trace`, `Step`, `Experience` |
| `itd/quality.py` | quality categories and scores, threshold θ |
| `itd/store/base.py` |  operator interface: `maintain`, `retrieve`, `retrieve_all`, experience cache |
| `itd/store/schema.sql`, `postgres.py` |  relational instantiation (Table 2), B-tree and HNSW indices |
| `itd/store/qdrant.py` |  vector-database instantiation (trace and template collections) |
| `itd/store/memory.py` | in-memory reference instantiation |
| `itd/strategies.py`, `prompts.py` |  materialization of experiences, synthesis prompts, injection wrappers |
| `itd/lifecycle.py` |  maintenance approaches and retrieval policies for one task |
| `itd/solver.py` |  zero-shot ReAct task solver |
| `itd/envs/` |  ALFWorld and WebArena adapters, templates, progress `m/M`; a toy environment |
| `itd/runner.py`, `cli.py` | experiment runner, raw and Retry baselines, result summaries |
| `itd/bench.py` |  operator latency on synthetic ITDs |
| `scripts/reproduce.sh` |  one function per table / figure |
| `tests/` |  operator semantics on every instantiation, Algorithm 1 |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[postgres,qdrant,embeddings,dev]"
docker compose up -d            # PostgreSQL 16 + pgvector 0.8.0, Qdrant 1.19.2
cp .env.example .env            # endpoints and API keys
pytest                          # set ITD_TEST_PG_DSN to include PostgreSQL
```

Models are declared in `configs/models.yaml` and reached through
OpenAI-compatible endpoints. Check that the `model` identifiers match your
deployment.

Benchmarks are optional extras:

- **ALFWorld**: `pip install -e ".[alfworld]"`, `alfworld-download`, set
  `ALFWORLD_DATA`, and copy ALFWorld's `configs/base_config.yaml` to
  `configs/alfworld_base.yaml`.
- **WebArena**: `pip install -e ".[webarena]"`, `playwright install chromium`,
  and the `WA_*` variables of a running WebArena instance. Trace quality on
  WebArena needs the intermediate checkpoints in
  `data/webarena_checkpoints.json` (format in
  `data/webarena_checkpoints.example.json`).

## Dry run

No API key, database or benchmark is needed: a toy environment, a mock LLM
that picks random admissible actions, and the in-memory store.

```bash
itd run --config configs/toy.yaml --mode raw --split train --tag populate
itd run --config configs/toy.yaml --approach hybrid --policy failure-triggered \
        --strategy combined --init-traces runs/toy/populate_seed0/traces.jsonl
itd summarize runs/toy/*
```

This only exercises the pipeline; its success rates mean nothing.

## Running the experiments

`itd run` runs one configuration: a YAML file (`configs/alfworld.yaml`,
`configs/webarena.yaml`) whose fields can be overridden by flags
(`itd run --help`). Each run writes `config.json`, `results.jsonl` (one record
per task, with timings) and `traces.jsonl` to `runs/<benchmark>/<name>_seed<seed>/`,
and uses its own ITD (PostgreSQL schema or Qdrant collections).

Offline and Hybrid ITDs are pre-populated from a JSONL file of traces, which
is simply the `traces.jsonl` of a run without experience reuse. Traces are
loaded through `Maintain`, so `k` and θ apply to the initial ITD too.


## Using the operators directly

```python
from itd.embeddings import SentenceEmbedder
from itd.model import Task, Trace, Step
from itd.quality import assign_quality
from itd.store import make_store

store = make_store("postgres", SentenceEmbedder(), "demo", dsn="postgresql://itd:itd@localhost/itd")
task = Task("map-7", "Map", "What is the duration required to walk from A to B, then drive from B to C?",
            "What is the duration required to walk from CMU to Apple Store Shadyside, then drive to Starbucks on Craig Street?")
trace = Trace(task.task_id, task.task_family, task.template_desc, actor="deepseek-v3.2",
              label="failure", reward=0.0, horizon=30, steps=[Step(0, "<page>", "click('12')")],
              progress=3 / 5)                    # m/M, reported by the environment

store.add_task(task)
store.maintain(assign_quality(trace, theta=0.5), k=1)     # Maintain(D, τ, k)
store.retrieve_all(task, n=1)                             # RetrieveAll(t, D, n)
```

