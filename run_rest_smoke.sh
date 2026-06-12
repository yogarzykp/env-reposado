#!/bin/bash
# =============================================================================
# ReST env-reposado smoke test (editable). Fill in the CONFIG block below on the
# GPU box, then run:   bash run_rest_smoke.sh
#
# It runs one (small) ReST training pass and, if HF_REPO/HF_TOKEN are set,
# uploads the merged model so you can inspect the training result on the Hub.
# =============================================================================

# ----------------------------- CONFIG (edit me) ------------------------------
# Base model: a local path or a HF model id.
MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"

# Validator env-server (comma-separated if several). Required for games.
ENVIRONMENT_SERVER_URLS="http://localhost:8000"

# Pick the game: set GAME, OR set TASK_ID (TASK_ID wins if non-empty).
#   goofspiel | liars_dice | leduc_poker | gin_rummy
GAME="leduc_poker"
TASK_ID=""

# Where the trained model + artefacts are written.
OUTPUT_DIR="/workspace/rest_smoke"

# ReST smoke knobs (small/fast; raise for a real run).
REST_ITERS=1
REST_SEEDS_PER_ITER=8
REST_N_PER_SEED=2
REST_TOTAL_SECONDS=1200
CFR_ITERS=3000
REST_SKILL_TASKS=0          # >0 adds the self-instruct skill stream

# HF upload to check training (leave HF_REPO empty to skip).
HF_REPO=""                  # e.g. yogarzykp/env-reposado-leduc-smoke
HF_TOKEN=""                 # your HF write token
HF_PRIVATE="1"              # 1 = private repo

# Python env that has vllm/trl/transformers. Leave empty to auto-detect a few
# common locations; set it explicitly if your deps live elsewhere.
VENV_PATH=""                # e.g. /workspace/.grpo_env
# -----------------------------------------------------------------------------

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH ENVIRONMENT_SERVER_URLS OUTPUT_DIR
export REST_ITERS REST_SEEDS_PER_ITER REST_N_PER_SEED REST_TOTAL_SECONDS CFR_ITERS REST_SKILL_TASKS
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-122}"
if [ -n "$TASK_ID" ]; then export TASK_ID; else export GAME; fi

# Activate the training venv (vllm/trl live there, not in the system python).
if [ -z "$VENV_PATH" ]; then
  for cand in /workspace/.grpo_env /workspace/venv "$HOME/.grpo_env" "$HOME/venv"; do
    [ -f "$cand/bin/activate" ] && VENV_PATH="$cand" && break
  done
fi
if [ -n "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/activate" ]; then
  echo "activating venv: $VENV_PATH"
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
fi

# Preflight: fail early with a clear message instead of a deep traceback.
if ! python3 -c "import vllm" >/dev/null 2>&1; then
  echo "ERROR: 'vllm' not importable by $(command -v python3)."
  echo "Fix one of:"
  echo "  1) set VENV_PATH in this script to the env that has vllm/trl/transformers, or"
  echo "  2) install deps:  pip install -r ${SCRIPT_DIR}/scripts/grpo_requirements.txt"
  echo "  3) run inside the standalone-text-trainer container (it ships the deps)"
  exit 1
fi

echo "=== ReST smoke: ${TASK_ID:-$GAME} | model=${MODEL_PATH} | iters=${REST_ITERS} ==="
python3 "${SCRIPT_DIR}/scripts/rest_trainer.py"

# --- optional: upload the merged model to the Hub -----------------------------
if [ -n "$HF_REPO" ] && [ -n "$HF_TOKEN" ]; then
  if [ -f "${OUTPUT_DIR}/config.json" ]; then
    echo "=== uploading ${OUTPUT_DIR} -> ${HF_REPO} ==="
    HF_TOKEN="$HF_TOKEN" HF_REPO="$HF_REPO" OUTPUT_DIR="$OUTPUT_DIR" HF_PRIVATE="$HF_PRIVATE" \
    python3 - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ["HF_REPO"]
api.create_repo(repo, repo_type="model", exist_ok=True,
                private=os.environ.get("HF_PRIVATE", "1") == "1")
api.upload_folder(folder_path=os.environ["OUTPUT_DIR"], repo_id=repo, repo_type="model",
                  ignore_patterns=["iter*_data/*", "iter*_sft/*"])
print(f"uploaded merged model to https://huggingface.co/{repo}")
PY
  else
    echo "WARN: ${OUTPUT_DIR}/config.json not found (training may have failed); skipping HF upload"
  fi
fi

echo "=== done. artefacts in ${OUTPUT_DIR} (history: rest_history.json) ==="
