#!/usr/bin/env bash
# Every experiment this repository supports, one function per experiment.
#
#   scripts/experiments.sh <step> [<step> ...]
#   scripts/experiments.sh populate_alfworld sweep_alfworld
#
# Results land in runs/<benchmark>/; aggregate them with
# `itd summarize runs/<benchmark>/*`.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS_ALFWORLD="0 1 2"            # ALFWorld is cheap enough to repeat
APPROACHES="offline online hybrid"
STRATEGIES="examples rules reflections combined"
POLICIES="on-demand failure-triggered"
WA_SUBSET="map,reddit,gitlab"     # 395 tasks, for the runs too costly to do in full

alf() { itd run --config configs/alfworld.yaml "$@"; }
web() { itd run --config configs/webarena.yaml "$@"; }

# -- Seed stores --------------------------------------------------------------
# Offline and Hybrid start from a store built by an agent that reused nothing,
# so these runs come first. Their traces.jsonl is the input to --init-traces.

populate_alfworld() {   # 180 training tasks, balanced over the six families
  mkdir -p data/traces
  for model in qwen2.5-72b qwen2.5-7b; do
    alf --mode raw --split train --solver $model --tag populate_$model
    cp runs/alfworld/populate_${model}_seed0/traces.jsonl data/traces/alfworld_$model.jsonl
  done
}

populate_webarena() {   # doubles as the no-reuse baseline
  mkdir -p data/traces
  web --mode raw
  cp runs/webarena/webarena_deepseek-v3.2_raw_seed0/traces.jsonl data/traces/webarena_deepseek-v3.2.jsonl
  web --mode raw --solver qwen2.5-7b --families $WA_SUBSET
  cp runs/webarena/webarena_qwen2.5-7b_raw_seed0/traces.jsonl data/traces/webarena_qwen2.5-7b.jsonl
}

# -- Does reuse help, and which combination? ----------------------------------
# Maintenance approach x reuse strategy x retrieval policy, against two
# baselines: no reuse at all, and a second unaided attempt (Retry), which
# separates the value of the retrieved experience from the value of the extra
# attempt that Failure-triggered retrieval spends.

sweep_alfworld() {
  for seed in $SEEDS_ALFWORLD; do
    alf --mode raw --seed $seed
    alf --mode retry --seed $seed
    for a in $APPROACHES; do for s in $STRATEGIES; do for p in $POLICIES; do
      alf --approach $a --strategy $s --policy $p --seed $seed --shuffle
    done; done; done
  done
}

sweep_webarena() {
  web --mode retry
  for a in $APPROACHES; do for s in $STRATEGIES; do for p in $POLICIES; do
    web --approach $a --strategy $s --policy $p
  done; done; done
}

sweep_webarena_qwen() { # the same sweep with a second task solver
  web --mode raw --solver qwen2.5-72b
  for a in $APPROACHES; do for s in $STRATEGIES; do
    web --approach $a --strategy $s --solver qwen2.5-72b
  done; done
}

# -- Does experience transfer between agents? ---------------------------------
# A small task solver reuses traces produced by a larger one, and experiences
# synthesized by either. Trace source and synthesizer vary independently.

transfer_alfworld() {
  for seed in $SEEDS_ALFWORLD; do
    alf --mode raw --solver qwen2.5-7b --seed $seed
    for traces in qwen2.5-7b qwen2.5-72b; do for synth in qwen2.5-7b qwen2.5-72b; do for a in offline hybrid; do
      alf --solver qwen2.5-7b --synthesizer $synth --init-traces data/traces/alfworld_$traces.jsonl \
          --approach $a --seed $seed --shuffle --tag transfer_${a}_traces-${traces}_synth-${synth}
    done; done; done
  done
}

transfer_webarena() {   # split by site with `itd summarize ... --by task_family`
  for traces in qwen2.5-7b deepseek-v3.2; do for synth in qwen2.5-7b deepseek-v3.2; do
    web --solver qwen2.5-7b --synthesizer $synth --init-traces data/traces/webarena_$traces.jsonl \
        --families $WA_SUBSET --tag transfer_traces-${traces}_synth-${synth}
  done; done
}

# -- What each knob is worth --------------------------------------------------
# All three hold the rest fixed at Offline / On-demand / Rules.

ablation_categories() { # which quality categories are worth retrieving
  for seed in $SEEDS_ALFWORLD; do
    for c in failure near-success success success,near-success near-success,failure success,failure; do
      alf --categories $c --seed $seed --tag ablation_categories_${c//,/+}
    done
  done
}

ablation_budgets() {    # how many traces to keep (k) and to retrieve (n)
  for seed in $SEEDS_ALFWORLD; do for b in 3 5 10; do
    alf --k $b --n $b --seed $seed --tag ablation_budget_$b
  done; done
}

ablation_theta() {      # where a failure becomes a near-success
  for t in 0.3 0.7; do
    for seed in $SEEDS_ALFWORLD; do alf --theta $t --seed $seed --tag ablation_theta_$t; done
    web --theta $t --families $WA_SUBSET --tag ablation_theta_$t
  done
}

# -- What the store itself costs ----------------------------------------------
bench() {               # Maintain and Retrieve latency, both backends
  itd bench --backends postgres,qdrant --dsn "$ITD_PG_DSN" --qdrant-url "$ITD_QDRANT_URL" --out runs/bench.csv
}

[ -f .env ] && set -a && . ./.env && set +a
for step in "$@"; do "$step"; done
