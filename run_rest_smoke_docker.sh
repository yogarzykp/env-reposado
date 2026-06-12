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
GPUS="all"                                  # or:  '"device=0"'  for one GPU

MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"     # HF id (downloaded) OR a local path
ENVIRONMENT_SERVER_URLS="http://localhost:8000"
GAME="leduc_poker"                          # goofspiel|liars_dice|leduc_poker|gin_rummy
TASK_ID=""                                  # optional; overrides GAME if set

# game       = full ReST loop (NEEDS a live env-server at ENVIRONMENT_SERVER_URLS)
# skill-only = validate generation + SFT + merge WITHOUT an env-server
SMOKE_MODE="game"

HOST_OUTPUT_DIR="$HOME/rest_smoke_out"       # trained model lands here (host)
HF_CACHE_DIR="$HOME/.cache/huggingface"      # persist model downloads

REST_ITERS=1
REST_SEEDS_PER_ITER=8
REST_N_PER_SEED=2
REST_TOTAL_SECONDS=1200
CFR_ITERS=3000
REST_SKILL_TASKS=0

HF_REPO=""                                   # e.g. yogarzykp/env-reposado-leduc-smoke
HF_TOKEN=""                                  # HF write token
HF_PRIVATE="1"
# -----------------------------------------------------------------------------

set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_OUTPUT="/workspace/rest_smoke"
mkdir -p "$HOST_OUTPUT_DIR" "$HF_CACHE_DIR"

if [ "$SMOKE_MODE" = "skill-only" ]; then REST_SKILL_ONLY=1; else REST_SKILL_ONLY=0; fi

export MODEL_PATH ENVIRONMENT_SERVER_URLS GAME TASK_ID
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
  -e HF_REPO -e HF_TOKEN -e HF_PRIVATE \
  -v "$REPO/scripts:/workspace/scripts" \
  -v "$HOST_OUTPUT_DIR:$CONTAINER_OUTPUT" \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  "${MODEL_MOUNT[@]}" \
  --entrypoint bash "$IMAGE" /workspace/scripts/_rest_container_run.sh

echo "=== done. artefacts on host: $HOST_OUTPUT_DIR ==="
