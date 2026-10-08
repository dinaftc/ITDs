# Interaction Trace Databases (ITDs)

Code for *Interaction Trace Databases for Agent Experience Reuse* (EDBT 2027).

An ITD stores the interaction traces produced by LLM agents and keeps their
storage, maintenance and retrieval separate from the way they are reused.
This repository contains the data model, the `Maintain` and `Retrieve`
operators over PostgreSQL/pgvector and Qdrant, the three reuse strategies
(Examples, Rules, Reflections, plus Combined), the On-demand and
Failure-triggered retrieval policies, and the scripts that run the paper's
experiments on WebArena and ALFWorld.

## Layout

| Path | Paper | Content |
|---|---|---|
| `itd/model.py` | §3.1 | `Task`, `Trace`, `Step`, `Experience` |
| `itd/quality.py` | §3.1 | quality categories and scores, threshold θ |
| `itd/store/base.py` | §3.3 | operator interface: `maintain`, `retrieve`, `retrieve_all`, experience cache |
| `itd/store/schema.sql`, `postgres.py` | §3.1, §4.8 | relational instantiation (Table 2), B-tree and HNSW indices |
| `itd/store/qdrant.py` | §3.1, §4.8 | vector-database instantiation (trace and template collections) |
| `itd/store/memory.py` | | in-memory reference instantiation |
| `itd/strategies.py`, `prompts.py` | §3.1, App. A | materialization of experiences, synthesis prompts, injection wrappers |
| `itd/lifecycle.py` | §3.2, Alg. 1 | maintenance approaches and retrieval policies for one task |
| `itd/solver.py` | §4.1 | zero-shot ReAct task solver |
| `itd/envs/` | §4.1 | ALFWorld and WebArena adapters, templates, progress `m/M`; a toy environment |
| `itd/runner.py`, `cli.py` | §4 | experiment runner, raw and Retry baselines, result summaries |
| `itd/bench.py` | §4.8 | operator latency on synthetic ITDs |
| `scripts/reproduce.sh` | §4 | one function per table / figure |
| `tests/` | | operator semantics on every instantiation, Algorithm 1 |

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

```bash
scripts/reproduce.sh populate_alfworld populate_webarena
```

| Paper | Step of `scripts/reproduce.sh` | Summary |
|---|---|---|
| Fig. 2, Table 3 | `rq1_rq2_alfworld`, `rq1_rq2_webarena`, `rq1_webarena_qwen` | `itd summarize runs/alfworld/*` |
| Table 4 | `rq1_rq2_webarena` | `itd summarize runs/webarena/* --by interaction_type` |
| Tables 5, 6 | `rq3_alfworld`, `rq3_webarena` | `--by task_family` for Table 6 |
| Table 9 | `ablation_categories` | |
| Fig. 3 | `ablation_budgets` | |
| Table 10 | `ablation_theta` | `--by task_family` |
| Figs. 4, 5 | `bench` | `runs/bench.csv` |
| Fig. 6 | any run | `total_s`, `llm_s`, `env_s`, `retrieve_s` in `results.jsonl` |

`itd summarize` reports the binomial standard error for a single run
(WebArena) and mean ± s.d. when several seeds of a configuration are given
(ALFWorld).

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

## Scope and limitations

- **ReMe** (Tables 7 and 8) is not included: the paper runs the authors'
  implementation with benchmark integrations.
- **WebArena checkpoints** are not included; without them every unsuccessful
  WebArena trace has progress 0 and falls in the `failure` partition.
- Table 4's split uses `interaction_type`, defined here as *information-seeking*
  when a task is evaluated by `string_match` only and *state-changing*
  otherwise.
- In the trace-count and budget benchmarks, all requests target stored
  templates; the template-count benchmark sends 40% of requests to unseen
  templates (`--fallback-frac`). The benchmark does not measure HNSW recall.
- No plotting scripts: runs produce JSONL and CSV.
