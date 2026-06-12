#!/bin/bash
# Inner runner executed INSIDE the standalone-text-trainer container by
# run_rest_smoke_docker.sh. Activates the trainer venv (vllm/trl live there),
# runs the ReST loop, then optionally uploads the merged model to the Hub.
# All inputs arrive via environment variables (docker -e ...).
set -e

source /workspace/.grpo_env/bin/activate
cd /workspace/scripts

# The image enables hf_transfer but the package isn't installed -> downloads
# crash. Disable it (normal, slightly slower download) unless hf_transfer exists.
if ! python3 -c "import hf_transfer" >/dev/null 2>&1; then
  export HF_HUB_ENABLE_HF_TRANSFER=0
fi

echo "python: $(command -v python3)"
python3 -c "import vllm, trl, transformers, peft; print('versions: vllm', vllm.__version__, '| trl', trl.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)"

[ -n "$TASK_ID" ] || unset TASK_ID    # empty TASK_ID -> fall back to GAME
python3 rest_trainer.py

if [ -n "$HF_REPO" ] && [ -n "$HF_TOKEN" ] && [ -f "${OUTPUT_DIR}/config.json" ]; then
  echo "=== uploading ${OUTPUT_DIR} -> ${HF_REPO} ==="
  python3 - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ["HF_REPO"]
api.create_repo(repo, repo_type="model", exist_ok=True,
                private=os.environ.get("HF_PRIVATE", "1") == "1")
api.upload_folder(folder_path=os.environ["OUTPUT_DIR"], repo_id=repo, repo_type="model",
                  ignore_patterns=["iter*_data/*", "iter*_sft/*"])
print("uploaded:", repo)
PY
elif [ -n "$HF_REPO" ]; then
  echo "WARN: ${OUTPUT_DIR}/config.json missing or HF_TOKEN empty; skipping HF upload"
fi
