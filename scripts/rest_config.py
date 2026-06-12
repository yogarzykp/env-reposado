"""Per-model-size hyperparameters for the iterative ReST loop.

This is the ReST analogue of a size-config table, but it parameterises a
*Grow/Improve* loop rather than a policy-gradient run: number of iterations, the
relative top-k% filter schedule, the MCTS opponent-strength curriculum, the
inner-SFT settings, and the LoRA rank. It is intentionally its own small scheme
(four coarse buckets) rather than the baseline's fine-grained table.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class RestBucket:
    name: str
    num_iters: int
    # Relative top-k% kept per iteration (shrinks as the policy improves).
    keep_fraction_schedule: List[float] = field(default_factory=lambda: [0.30, 0.20, 0.10])
    # MCTS opponent strength per iteration (cold-start curriculum: weak -> strong).
    opponent_sims_schedule: List[int] = field(default_factory=lambda: [5, 15, 50])
    temperature: float = 1.0           # Grow sampling temperature (best-of-N)
    n_per_seed: int = 4                 # episodes per seed (best-of-N)
    seeds_per_iter: int = 256           # distinct game seeds per Grow phase
    inner_sft_lr: float = 1e-4
    inner_sft_epochs: int = 1
    lora_r: int = 32
    lora_alpha: int = 64
    gpu_count: int = 1
    max_new_tokens: int = 256
    # If iteration-1 valid-action rate is below this, label with cold-start
    # template Thoughts instead of model rationalization.
    cold_start_valid_threshold: float = 0.5


# Four buckets by parameter count (billions). Smaller models get more iterations
# and a gentler opponent ramp; larger models fewer iterations (each costs more).
_BUCKETS = [
    RestBucket(
        name="small_lt2b", num_iters=4,
        keep_fraction_schedule=[0.35, 0.25, 0.15, 0.10],
        opponent_sims_schedule=[5, 10, 25, 50],
        temperature=1.1, n_per_seed=6, seeds_per_iter=384,
        inner_sft_lr=1.5e-4, inner_sft_epochs=1, lora_r=32, lora_alpha=64,
        gpu_count=1, max_new_tokens=256, cold_start_valid_threshold=0.4,
    ),
    RestBucket(
        name="mid_2to9b", num_iters=3,
        keep_fraction_schedule=[0.30, 0.20, 0.10],
        opponent_sims_schedule=[8, 20, 50],
        temperature=1.0, n_per_seed=4, seeds_per_iter=256,
        inner_sft_lr=1e-4, inner_sft_epochs=1, lora_r=32, lora_alpha=64,
        gpu_count=1, max_new_tokens=256, cold_start_valid_threshold=0.5,
    ),
    RestBucket(
        name="large_9to40b", num_iters=2,
        keep_fraction_schedule=[0.25, 0.12],
        opponent_sims_schedule=[15, 50],
        temperature=0.9, n_per_seed=4, seeds_per_iter=192,
        inner_sft_lr=7e-5, inner_sft_epochs=1, lora_r=16, lora_alpha=32,
        gpu_count=2, max_new_tokens=256, cold_start_valid_threshold=0.6,
    ),
    RestBucket(
        name="xl_gt40b", num_iters=2,
        keep_fraction_schedule=[0.20, 0.10],
        opponent_sims_schedule=[20, 50],
        temperature=0.8, n_per_seed=3, seeds_per_iter=128,
        inner_sft_lr=5e-5, inner_sft_epochs=1, lora_r=16, lora_alpha=32,
        gpu_count=4, max_new_tokens=256, cold_start_valid_threshold=0.6,
    ),
]


def estimate_param_billions(model_name_or_path: str) -> float:
    """Best-effort parameter count (in billions) parsed from a model name."""
    if not model_name_or_path:
        return 7.0
    text = model_name_or_path.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", text)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*m\b", text)
    if m:
        return float(m.group(1)) / 1000.0
    return 7.0


def get_rest_config(param_billions: float) -> RestBucket:
    """Pick the bucket for a parameter count. Env vars override key knobs."""
    if param_billions < 2:
        bucket = _BUCKETS[0]
    elif param_billions < 9:
        bucket = _BUCKETS[1]
    elif param_billions < 40:
        bucket = _BUCKETS[2]
    else:
        bucket = _BUCKETS[3]
    return _apply_env_overrides(bucket)


def _apply_env_overrides(bucket: RestBucket) -> RestBucket:
    """Allow runtime overrides without editing the table (one knob per env)."""
    if os.environ.get("REST_ITERS"):
        bucket.num_iters = int(os.environ["REST_ITERS"])
    if os.environ.get("REST_TEMPERATURE"):
        bucket.temperature = float(os.environ["REST_TEMPERATURE"])
    if os.environ.get("REST_N_PER_SEED"):
        bucket.n_per_seed = int(os.environ["REST_N_PER_SEED"])
    if os.environ.get("REST_SEEDS_PER_ITER"):
        bucket.seeds_per_iter = int(os.environ["REST_SEEDS_PER_ITER"])
    if os.environ.get("REST_SFT_LR"):
        bucket.inner_sft_lr = float(os.environ["REST_SFT_LR"])
    if os.environ.get("REST_SFT_EPOCHS"):
        bucket.inner_sft_epochs = int(os.environ["REST_SFT_EPOCHS"])
    if os.environ.get("REST_LORA_R"):
        bucket.lora_r = int(os.environ["REST_LORA_R"])
    return bucket


def schedule_at(schedule: List, i: int):
    """Value for iteration ``i``; clamps to the last entry when the loop runs
    longer than the schedule."""
    if not schedule:
        raise ValueError("empty schedule")
    return schedule[i] if i < len(schedule) else schedule[-1]


def _selftest() -> None:
    assert estimate_param_billions("Qwen2.5-7B-Instruct") == 7.0
    assert estimate_param_billions("Qwen3-0.6B") == 0.6
    assert estimate_param_billions("foo-130M") == 0.13
    assert get_rest_config(0.6).name == "small_lt2b"
    assert get_rest_config(7).name == "mid_2to9b"
    assert get_rest_config(32).name == "large_9to40b"
    assert get_rest_config(70).name == "xl_gt40b"
    assert schedule_at([5, 15, 50], 9) == 50
    print("rest_config selftest OK")


if __name__ == "__main__":
    _selftest()
