#!/bin/bash
# Smoke-test examples for env-reposado (ReST env trainer).
#
# Usage:  bash scripts/examples.sh <mode>
#
#   selftest     offline module selftests (no GPU, no env-server) -- run first
#   leduc_poker | goofspiel | liars_dice | gin_rummy
#                single-game smoke (needs env-server + model + GPU)
#   all-games    run all four games in sequence
#   skill        leduc smoke WITH the self-instruct skill stream enabled
#   skill-only   skill stream only (needs model + GPU, NO env-server)
#
# Set these before the GPU smokes:
#   export MODEL_PATH=/path/to/small-model
#   export ENVIRONMENT_SERVER_URLS=http://host:port
#
# Smoke knobs default to small/fast values; override by exporting first.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-selftest}"

export REST_ITERS="${REST_ITERS:-1}"
export REST_SEEDS_PER_ITER="${REST_SEEDS_PER_ITER:-8}"
export REST_N_PER_SEED="${REST_N_PER_SEED:-2}"
export REST_TOTAL_SECONDS="${REST_TOTAL_SECONDS:-900}"
export CFR_ITERS="${CFR_ITERS:-3000}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/rest_smoke}"

run_game () {  # $1 = GAME name
  : "${MODEL_PATH:?set MODEL_PATH}"
  : "${ENVIRONMENT_SERVER_URLS:?set ENVIRONMENT_SERVER_URLS}"
  echo "=== smoke: game=$1 (iters=${REST_ITERS}, seeds=${REST_SEEDS_PER_ITER}) ==="
  GAME="$1" OUTPUT_DIR="${OUTPUT_DIR}/$1" bash "${SCRIPT_DIR}/run_rest_env.sh"
}

case "$MODE" in
  selftest)
    cd "$SCRIPT_DIR"
    for m in cfr_leduc shaping game_spec rest_config; do echo "--- $m ---"; python3 "${m}.py"; done
    echo "--- rest_trainer ---"; python3 rest_trainer.py --selftest
    for m in rollout_collector trajectory_filter cot_synthesizer skill_selfinstruct; do
      echo "--- selfplay.$m ---"; python3 -m "selfplay.${m}"
    done
    echo "selftest suite OK"
    ;;

  leduc_poker|goofspiel|liars_dice|gin_rummy)
    run_game "$MODE"
    ;;

  all-games)
    for g in goofspiel liars_dice leduc_poker gin_rummy; do run_game "$g"; done
    ;;

  skill)
    : "${MODEL_PATH:?set MODEL_PATH}"
    : "${ENVIRONMENT_SERVER_URLS:?set ENVIRONMENT_SERVER_URLS}"
    echo "=== smoke: leduc + skill stream (REST_SKILL_TASKS) ==="
    REST_SKILL_TASKS="${REST_SKILL_TASKS:-8}" GAME=leduc_poker \
      OUTPUT_DIR="${OUTPUT_DIR}/leduc_skill" bash "${SCRIPT_DIR}/run_rest_env.sh"
    ;;

  skill-only)
    : "${MODEL_PATH:?set MODEL_PATH}"
    echo "=== smoke: skill stream only (no env-server) ==="
    cd "$SCRIPT_DIR"
    python3 - <<'PY'
import os
from rest_config import get_rest_config, estimate_param_billions
from rest_trainer import build_vllm_generate_fn
from selfplay.skill_selfinstruct import collect_skill_samples

model = os.environ["MODEL_PATH"]
cfg = get_rest_config(estimate_param_billions(model))
gen = build_vllm_generate_fn(model, None, cfg)
n = int(os.environ.get("REST_SKILL_TASKS", "8"))
out = collect_skill_samples(gen, n_tasks=n)
print(f"[skill-only] solved {len(out)}/{n} tasks")
PY
    ;;

  *)
    echo "unknown mode: $MODE"
    echo "modes: selftest | leduc_poker | goofspiel | liars_dice | gin_rummy | all-games | skill | skill-only"
    exit 1
    ;;
esac
