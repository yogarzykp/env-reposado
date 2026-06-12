#!/bin/bash
# Entrypoint for the ReST env trainer (env-reposado).
# Runs scripts/rest_trainer.py, which reads model/output/env-server/task from
# CLI args or env vars. Use directly for local/GPU smoke, or via text_trainer
# with REST_TRAINER=1.
set -e

# bitsandbytes must target the container CUDA (image max cu122); the in-Python
# guard in train code also sets this for torchrun workers.
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-122}"

# Some base stacks expect a running redis; start it if available (no-op else).
if command -v redis-server >/dev/null 2>&1; then
  redis-server --daemonize yes || true
  sleep 3
fi

echo "***** Running ReST env trainer"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d /workspace/.grpo_env ]; then
  source /workspace/.grpo_env/bin/activate
fi

python3 "${SCRIPT_DIR}/rest_trainer.py" "$@"

if [ -d /workspace/.grpo_env ]; then
  deactivate || true
fi
