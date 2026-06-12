#!/bin/bash
# =============================================================================
# ReST smoke INSIDE the standalone-text-trainer container (has vllm/trl/etc).
# Edit the CONFIG block, then:   bash run_rest_smoke_docker.sh
#
# It mounts your latest scripts/ over the image, runs one small ReST pass, and
# (optionally) uploads the merged model to the Hub. Output is persisted on host.
# =============================================================================

# ----------------------------- CONFIG (edit me) ------------------------------
IMAGE="standalone-text-trainer:latest"
GPUS="all"                                  # docker --gpus value
# Pin the process to ONE GPU: HF Trainer auto-uses DataParallel across visible
# GPUs, which crashes with NCCL on PCIe/no-NVLink boxes. Single GPU avoids it.
CUDA_VISIBLE_DEVICES="0"

MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"     # HF id (downloaded) OR a local path
ENVIRONMENT_SERVER_URLS="http://localhost:8000"
GAME="leduc_poker"                          # goofspiel|liars_dice|leduc_poker|gin_rummy
TASK_ID=""                                  # optional; overrides GAME if set

# game       = full ReST loop (NEEDS a live env-server at ENVIRONMENT_SERVER_URLS)
# skill-only = validate generation + SFT + merge WITHOUT an env-server
SMOKE_MODE="game"

# Auto start+stop the MCTS env-server for game mode (no separate commands needed).
# Set to 0 to use your own ENVIRONMENT_SERVER_URLS above instead.
AUTO_ENV_SERVER="1"
NUM_SERVERS="1"
ENV_BASE_PORT="8000"
MCTS_IMAGE="diagonalge/mcts-api:latest"

HOST_OUTPUT_DIR="$HOME/rest_smoke_out"       # trained model lands here (host)
HF_CACHE_DIR="$HOME/.cache/huggingface"      # persist model downloads

REST_ITERS=1
REST_SEEDS_PER_ITER=8
REST_N_PER_SEED=2
REST_TOTAL_SECONDS=1200
CFR_ITERS=3000
REST_SKILL_TASKS=0

# HF upload. Leave HF_USERNAME empty to skip. The repo name is auto-built as
#   <prefix>-<game>-<date>_<model>   (like env-purpleboost's expected-repo-name).
HF_USERNAME=""                               # e.g. yogarzykp  (empty = skip upload)
HF_TOKEN=""                                  # HF write token
HF_PRIVATE="1"
HF_REPO_PREFIX="env-reposado"
# -----------------------------------------------------------------------------

set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_OUTPUT="/workspace/rest_smoke"
mkdir -p "$HOST_OUTPUT_DIR" "$HF_CACHE_DIR"

if [ "$SMOKE_MODE" = "skill-only" ]; then REST_SKILL_ONLY=1; else REST_SKILL_ONLY=0; fi

# --- auto-managed env-server (game mode): started before, stopped on exit ---
STARTED_ENV=0
cleanup_env() {
  if [ "$STARTED_ENV" = "1" ]; then
    for i in $(seq 0 $((NUM_SERVERS - 1))); do docker rm -f "mcts-env-$i" >/dev/null 2>&1 || true; done
    echo "=== stopped env-server(s) ==="
  fi
}
trap cleanup_env EXIT

if [ "$SMOKE_MODE" = "game" ] && [ "$AUTO_ENV_SERVER" = "1" ]; then
  echo "=== starting env-server(s): $MCTS_IMAGE ==="
  docker pull "$MCTS_IMAGE" || echo "WARN: pull failed (using local image if present)"
  urls=()
  for i in $(seq 0 $((NUM_SERVERS - 1))); do
    port=$((ENV_BASE_PORT + i)); name="mcts-env-$i"
    docker rm -f "$name" >/dev/null 2>&1 || true
    docker run -d --name "$name" -p "${port}:8000" "$MCTS_IMAGE" >/dev/null
    urls+=("http://localhost:${port}")
  done
  STARTED_ENV=1
  ENVIRONMENT_SERVER_URLS=$(IFS=,; echo "${urls[*]}")
  echo "  ENVIRONMENT_SERVER_URLS=$ENVIRONMENT_SERVER_URLS"
  if command -v curl >/dev/null 2>&1; then
    echo "  waiting for env-server readiness..."
    for _ in $(seq 1 30); do
      if curl -s -o /dev/null -X POST "http://localhost:${ENV_BASE_PORT}/reset" \
           -H 'Content-Type: application/json' \
           -d '{"task_id":200000001,"seed":1,"opponent":"mcts","mcts_max_simulations":5,"mcts_num_rollouts":1}'; then
        echo "  env-server ready"; break
      fi
      sleep 2
    done
  else
    echo "  curl not found; waiting 10s for warmup"; sleep 10
  fi
fi

# Build a unique HF repo id: <username>/<prefix>-<game>-<date>_<model>.
if [ -n "$HF_USERNAME" ]; then
  MODEL_SAFE=$(echo "$MODEL_PATH" | sed 's#/#_#g')
  EXPECTED_REPO_NAME="${HF_REPO_PREFIX}-${TASK_ID:-$GAME}-$(date +%Y%m%d-%H%M)_${MODEL_SAFE}"
  HF_REPO="${HF_USERNAME}/${EXPECTED_REPO_NAME}"
  echo "=== HF target repo: $HF_REPO ==="
else
  HF_REPO=""
fi

export MODEL_PATH ENVIRONMENT_SERVER_URLS GAME TASK_ID CUDA_VISIBLE_DEVICES
export REST_ITERS REST_SEEDS_PER_ITER REST_N_PER_SEED REST_TOTAL_SECONDS CFR_ITERS REST_SKILL_TASKS REST_SKILL_ONLY
export HF_REPO HF_TOKEN HF_PRIVATE

# Mount a local model dir read-only if MODEL_PATH points at one.
MODEL_MOUNT=()
[ -d "$MODEL_PATH" ] && MODEL_MOUNT=(-v "$MODEL_PATH:$MODEL_PATH:ro")

echo "=== ReST smoke (docker): ${TASK_ID:-$GAME} | image=$IMAGE | gpus=$GPUS ==="
docker run --rm -i --gpus "$GPUS" --network host \
  -e MODEL_PATH -e ENVIRONMENT_SERVER_URLS -e GAME -e TASK_ID \
  -e OUTPUT_DIR="$CONTAINER_OUTPUT" \
  -e REST_ITERS -e REST_SEEDS_PER_ITER -e REST_N_PER_SEED -e REST_TOTAL_SECONDS \
  -e CFR_ITERS -e REST_SKILL_TASKS -e REST_SKILL_ONLY -e BNB_CUDA_VERSION=122 \
  -e CUDA_VISIBLE_DEVICES \
  -e HF_REPO -e HF_TOKEN -e HF_PRIVATE \
  -v "$REPO/scripts:/workspace/scripts" \
  -v "$HOST_OUTPUT_DIR:$CONTAINER_OUTPUT" \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  "${MODEL_MOUNT[@]}" \
  --entrypoint bash "$IMAGE" /workspace/scripts/_rest_container_run.sh

echo "=== done. artefacts on host: $HOST_OUTPUT_DIR ==="
