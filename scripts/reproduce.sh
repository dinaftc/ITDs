#!/usr/bin/env bash
# Reproduces the experiments of the paper, one function per table / figure.
#   scripts/reproduce.sh <step> [<step> ...]        e.g. scripts/reproduce.sh populate_alfworld rq1_rq2_alfworld
# Results land in runs/<benchmark>/; aggregate them with `itd summarize runs/<benchmark>/*`.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS_ALFWORLD="0 1 2"          # ALFWorld: mean ± s.d. over three runs
APPROACHES="offline online hybrid"
STRATEGIES="examples rules reflections combined"
POLICIES="on-demand failure-triggered"
WA_SUBSET="Map,Reddit,GitLab"   # 395 tasks used for RQ3 and the theta ablation

alf() { itd run --config configs/alfworld.yaml "$@"; }
web() { itd run --config configs/webarena.yaml "$@"; }

# -- initial ITDs (Section 4.1, "ITD parameters") -----------------------------
populate_alfworld() {   # 180 training tasks, no experience reuse
  mkdir -p data/traces
  for model in qwen2.5-72b qwen2.5-7b; do
    alf --mode raw --split train --solver $model --tag populate_$model
    cp runs/alfworld/populate_${model}_seed0/traces.jsonl data/traces/alfworld_$model.jsonl
  done
}
populate_webarena() {   # raw DeepSeek-V3.2 traces; also the raw-agent baseline of Tables 3 and 7
  mkdir -p data/traces
  web --mode raw
  cp runs/webarena/webarena_deepseek-v3.2_raw_seed0/traces.jsonl data/traces/webarena_deepseek-v3.2.jsonl
  web --mode raw --solver qwen2.5-7b --families $WA_SUBSET
  cp runs/webarena/webarena_qwen2.5-7b_raw_seed0/traces.jsonl data/traces/webarena_qwen2.5-7b.jsonl
}

# -- RQ1 / RQ2: Figure 2, Tables 3 and 4 --------------------------------------
rq1_rq2_alfworld() {
  for seed in $SEEDS_ALFWORLD; do
    alf --mode raw --seed $seed
    alf --mode retry --seed $seed
    for a in $APPROACHES; do for s in $STRATEGIES; do for p in $POLICIES; do
      alf --approach $a --strategy $s --policy $p --seed $seed --shuffle
    done; done; done
  done
}
rq1_rq2_webarena() {    # Table 4: itd summarize runs/webarena/* --by interaction_type
  web --mode retry
  for a in $APPROACHES; do for s in $STRATEGIES; do for p in $POLICIES; do
    web --approach $a --strategy $s --policy $p
  done; done; done
}
rq1_webarena_qwen() {   # RQ1 also evaluates Qwen2.5-72B on WebArena
  web --mode raw --solver qwen2.5-72b
  for a in $APPROACHES; do for s in $STRATEGIES; do
    web --approach $a --strategy $s --solver qwen2.5-72b
  done; done
}

# -- RQ3: Tables 5 and 6 (Qwen2.5-7B task solver, On-demand, Rules) ------------
rq3_alfworld() {
  for seed in $SEEDS_ALFWORLD; do
    alf --mode raw --solver qwen2.5-7b --seed $seed
    for traces in qwen2.5-7b qwen2.5-72b; do for synth in qwen2.5-7b qwen2.5-72b; do for a in offline hybrid; do
      alf --solver qwen2.5-7b --synthesizer $synth --init-traces data/traces/alfworld_$traces.jsonl \
          --approach $a --seed $seed --shuffle --tag rq3_${a}_traces-${traces}_synth-${synth}
    done; done; done
  done
}
rq3_webarena() {        # per-family columns: itd summarize ... --by task_family
  for traces in qwen2.5-7b deepseek-v3.2; do for synth in qwen2.5-7b deepseek-v3.2; do
    web --solver qwen2.5-7b --synthesizer $synth --init-traces data/traces/webarena_$traces.jsonl \
        --families $WA_SUBSET --tag rq3_traces-${traces}_synth-${synth}
  done; done
}

# -- Ablations: Table 9, Figure 3, Table 10 (Offline, On-demand, Rules) --------
ablation_categories() {
  for seed in $SEEDS_ALFWORLD; do
    for c in failure near-success success success,near-success near-success,failure success,failure; do
      alf --categories $c --seed $seed --tag ablation_categories_${c//,/+}
    done
  done
}
ablation_budgets() {
  for seed in $SEEDS_ALFWORLD; do for b in 3 5 10; do
    alf --k $b --n $b --seed $seed --tag ablation_budget_$b
  done; done
}
ablation_theta() {
  for t in 0.3 0.7; do
    for seed in $SEEDS_ALFWORLD; do alf --theta $t --seed $seed --tag ablation_theta_$t; done
    web --theta $t --families $WA_SUBSET --tag ablation_theta_$t
  done
}

# -- Operator efficiency: Figures 4 and 5 --------------------------------------
bench() {
  itd bench --backends postgres,qdrant --dsn "$ITD_PG_DSN" --qdrant-url "$ITD_QDRANT_URL" --out runs/bench.csv
}

[ -f .env ] && set -a && . ./.env && set +a
for step in "$@"; do "$step"; done
