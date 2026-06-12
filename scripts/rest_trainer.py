"""Improve phase + outer loop: iterative ReST^EM controller for env-reposado.

One ReST iteration:

    Grow      collect self-play episodes (best-of-N, opponent curriculum)
    Filter    hard-filter (gate raw outcome -> rank shaped -> prune turns)
    Label     STaR rationalize the survivors into {messages} SFT samples
    Improve   re-anchor SFT: fine-tune the BASE model on the *accumulated* data

The re-anchor (always start the SFT from the base weights on the growing dataset,
ReST^EM, Singh et al. 2023 arXiv 2312.06585) is what controls drift across
iterations -- it plays the role KL/beta plays in GRPO. The *generator* still
improves each round because it loads the previous iteration's adapter.

Heavy backends (vLLM, TRL/torch) are lazy-imported inside their functions so this
module imports cleanly. The loop itself takes its backends by injection, so the
orchestration logic (budget, kill-criterion, accumulation, re-anchor) is testable
without a GPU.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from rest_config import RestBucket, get_rest_config, estimate_param_billions, schedule_at
from game_spec import GameSpec, get_spec, get_spec_by_task_id
from selfplay.rollout_collector import Episode, GenerateFn, play_episode
from selfplay.trajectory_filter import filter_and_extract
from selfplay.cot_synthesizer import synthesize
from selfplay.skill_selfinstruct import collect_skill_samples

LEDUC_GAME_ID = 200000000


# --------------------------------------------------------------------------- #
# Budget controller (wall-clock aware; iteration 1 gets a heavier slice)
# --------------------------------------------------------------------------- #


class RestBudget:
    def __init__(self, total_seconds: float, first_iter_weight: float = 2.0):
        self.total_seconds = float(total_seconds)
        self.first_iter_weight = first_iter_weight
        self.start = time.monotonic()

    def _weights(self, num_iters: int) -> List[float]:
        return [self.first_iter_weight] + [1.0] * (num_iters - 1) if num_iters > 0 else []

    def iteration_deadline(self, i: int, num_iters: int) -> float:
        """Absolute monotonic deadline by which iteration ``i`` should finish."""
        weights = self._weights(num_iters)
        total_w = sum(weights) or 1.0
        used = sum(weights[: i + 1]) / total_w
        return self.start + self.total_seconds * used

    def remaining(self) -> float:
        return self.total_seconds - (time.monotonic() - self.start)


# --------------------------------------------------------------------------- #
# Kill-criterion: stop the loop early if win-rate stops improving
# --------------------------------------------------------------------------- #


class KillCriterion:
    def __init__(self, window: int = 3, min_improvement: float = 0.01):
        self.window = window
        self.min_improvement = min_improvement
        self.win_rates: List[float] = []

    def update(self, win_rate: float) -> None:
        self.win_rates.append(win_rate)

    def should_pivot(self) -> bool:
        if len(self.win_rates) < self.window:
            return False
        recent = self.win_rates[-self.window :]
        return (recent[-1] - recent[0]) <= self.min_improvement


# --------------------------------------------------------------------------- #
# Stats + helpers
# --------------------------------------------------------------------------- #


@dataclass
class IterStats:
    iteration: int
    win_rate: float
    valid_rate: float
    n_episodes: int
    n_new_samples: int
    n_total_samples: int
    adapter_path: Optional[str]


def episode_stats(episodes: List[Episode]) -> Dict[str, float]:
    if not episodes:
        return {"win_rate": 0.0, "valid_rate": 0.0}
    wins = sum(1 for ep in episodes if ep.won)
    total_turns = sum(ep.num_turns for ep in episodes) or 1
    valid_turns = sum(1 for ep in episodes for t in ep.turns if t.valid)
    return {"win_rate": wins / len(episodes), "valid_rate": valid_turns / total_turns}


def sample_seeds(iteration: int, n: int, base_seed: int = 12345) -> List[int]:
    rng = random.Random((base_seed * 1000003) ^ (iteration + 1))
    return [rng.randint(0, 2**31 - 1) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Default backends (lazy-imported; not exercised by the selftest)
# --------------------------------------------------------------------------- #


def build_vllm_generate_fn(base_model_path: str, adapter_path: Optional[str],
                           cfg: RestBucket) -> GenerateFn:
    """Build a chat generate_fn backed by vLLM, applying the model chat template.

    The generator is base + the previous iteration's LoRA (if any); the SFT step
    always re-anchors to the base, so this is the only place the improving policy
    is loaded for sampling.
    """
    from transformers import AutoTokenizer  # lazy
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    # Leave GPU headroom so the inner SFT can share the same device after gen.
    llm = LLM(model=base_model_path, enable_lora=adapter_path is not None,
              max_lora_rank=cfg.lora_r, dtype="bfloat16",
              gpu_memory_utilization=float(os.environ.get("REST_VLLM_GPU_UTIL", "0.45")))
    lora_req = LoRARequest("rest_adapter", 1, adapter_path) if adapter_path else None

    def generate_fn(message_batches, n: int = 1, temperature: float = 1.0):
        prompts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in message_batches
        ]
        params = SamplingParams(n=n, temperature=temperature, max_tokens=cfg.max_new_tokens)
        outputs = llm.generate(prompts, params, lora_request=lora_req)
        return [[o.text for o in out.outputs] for out in outputs]

    return generate_fn


def to_hf_dataset(samples: List[Dict[str, object]], out_dir: str,
                  val_fraction: float = 0.05) -> str:
    """Persist accumulated {messages} samples as an HF DatasetDict; return path."""
    from datasets import Dataset, DatasetDict  # lazy

    os.makedirs(out_dir, exist_ok=True)
    n_val = max(1, int(len(samples) * val_fraction)) if len(samples) > 20 else 0
    train, val = samples[n_val:], samples[:n_val]
    ds = {"train": Dataset.from_list(train)}
    if val:
        ds["validation"] = Dataset.from_list(val)
    DatasetDict(ds).save_to_disk(out_dir)
    return out_dir


def run_inner_sft(base_model_path: str, dataset_path: str, out_dir: str,
                  cfg: RestBucket) -> str:
    """Re-anchored inner SFT: fresh LoRA on the BASE model, assistant-only loss.

    Always starts from ``base_model_path`` (never the previous adapter) so drift
    does not compound across ReST iterations.

    Target stack: the env/GRPO container (trl==0.27.0, transformers==4.57.5,
    peft==0.18.1). ``assistant_only_loss`` is the one knob to confirm at smoke;
    if unsupported, fall back to manual label masking of non-assistant spans.
    """
    import torch  # lazy
    from datasets import load_from_disk
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    os.makedirs(out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    dataset = load_from_disk(dataset_path)

    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM")

    # assistant_only_loss needs a chat template with a {% generation %} block to
    # build the assistant mask. Qwen/Llama default templates lack it -> fall back
    # to full-sequence SFT (standard, safe) instead of crashing.
    use_assistant_only = "endgeneration" in (tokenizer.chat_template or "")
    if not use_assistant_only:
        print("[ReST] chat template has no {% generation %} mask; "
              "assistant_only_loss disabled (full-sequence SFT)")

    sft_args = SFTConfig(
        output_dir=out_dir,
        num_train_epochs=cfg.inner_sft_epochs,
        learning_rate=cfg.inner_sft_lr,
        per_device_train_batch_size=int(os.environ.get("REST_SFT_BATCH", "4")),
        gradient_accumulation_steps=int(os.environ.get("REST_SFT_GA", "4")),
        bf16=True,
        logging_steps=int(os.environ.get("REST_SFT_LOG_STEPS") or "5"),
        save_strategy="no",
        assistant_only_loss=use_assistant_only,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model, args=sft_args, train_dataset=dataset["train"],
        processing_class=tokenizer, peft_config=lora,
    )
    trainer.train()
    adapter_dir = os.path.join(out_dir, "adapter")
    trainer.save_model(adapter_dir)
    return adapter_dir


def merge_and_save_final(base_model_path: str, adapter_path: Optional[str],
                         out_dir: str, log_fn: Callable[[str], None] = print) -> Optional[str]:
    """Finalise: merge the final LoRA adapter into the base and save a full model
    to ``out_dir`` so the validator can load/serve it directly.

    This is the env-reposado finalisation step (not a standalone merge utility):
    the loop serves LoRA adapters directly via vLLM between iterations, so a merge
    is only needed once, at the end, for the uploaded artefact.
    """
    if not adapter_path:
        log_fn("[ReST] no adapter to merge; skipping final merge")
        return None
    import torch  # lazy
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log_fn(f"[ReST] merging final adapter {adapter_path} -> {out_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def collect_until_deadline(endpoint: str, generate_fn: GenerateFn, seeds: List[int],
                           cfg: RestBucket, opponent_sims: int, deadline: float,
                           spec: GameSpec, log_fn: Callable[[str], None] = print) -> List[Episode]:
    """Best-of-N self-play that stops at the wall-clock deadline.

    Logs running win/valid rate every REST_GROW_LOG_EVERY episodes so the (long)
    Grow phase is observable rather than silent.
    """
    log_every = int(os.environ.get("REST_GROW_LOG_EVERY") or "16")
    episodes: List[Episode] = []
    for seed in seeds:
        if time.monotonic() >= deadline:
            break
        for _ in range(cfg.n_per_seed):
            if time.monotonic() >= deadline:
                break
            episodes.append(
                play_episode(endpoint, spec.game_id, opponent_sims, generate_fn,
                             temperature=cfg.temperature, seed=seed,
                             feature_fn=spec.features_fn, system_prompt=spec.system_prompt)
            )
            n = len(episodes)
            if log_every > 0 and n % log_every == 0:
                s = episode_stats(episodes)
                remaining = max(0.0, deadline - time.monotonic())
                log_fn(f"  [grow] episodes={n} wins={sum(e.won for e in episodes)} "
                       f"win_rate={s['win_rate']:.3f} valid_rate={s['valid_rate']:.3f} "
                       f"t_left={remaining:.0f}s")
    log_fn(f"  [grow] collected {len(episodes)} episodes (sims={opponent_sims})")
    return episodes


# --------------------------------------------------------------------------- #
# The ReST outer loop (backends injected for testability)
# --------------------------------------------------------------------------- #

GenFactory = Callable[[str, Optional[str], RestBucket], GenerateFn]
SftFn = Callable[[str, str, str, RestBucket], str]
DatasetFn = Callable[[List[Dict[str, object]], str], str]
CollectFn = Callable[..., List[Episode]]
MergeFn = Callable[..., Optional[str]]


def rest_loop(
    cfg: RestBucket,
    endpoint: str,
    base_model_path: str,
    out_dir: str,
    spec: Optional[GameSpec] = None,
    gen_factory: GenFactory = build_vllm_generate_fn,
    sft_fn: SftFn = run_inner_sft,
    collect_fn: CollectFn = collect_until_deadline,
    dataset_fn: DatasetFn = to_hf_dataset,
    merge_fn: MergeFn = merge_and_save_final,
    budget: Optional[RestBudget] = None,
    kill: Optional[KillCriterion] = None,
    base_seed: int = 12345,
    skill_tasks_per_iter: int = 0,
    log_fn: Callable[[str], None] = print,
) -> Dict[str, object]:
    spec = spec or get_spec("leduc_poker")
    budget = budget or RestBudget(float(os.environ.get("REST_TOTAL_SECONDS") or "3600"))
    kill = kill or KillCriterion()
    accumulated: List[Dict[str, object]] = []
    adapter_path: Optional[str] = None
    history: List[IterStats] = []

    for it in range(cfg.num_iters):
        sims = schedule_at(cfg.opponent_sims_schedule, it)
        keep_frac = schedule_at(cfg.keep_fraction_schedule, it)
        deadline = budget.iteration_deadline(it, cfg.num_iters)

        t0 = time.monotonic()
        log_fn(f"[ReST iter {it}] grow: opponent_sims={sims} seeds={cfg.seeds_per_iter} keep={keep_frac}")

        # Grow: generator is base + previous adapter (the improving policy).
        gen_fn = gen_factory(base_model_path, adapter_path, cfg)
        seeds = sample_seeds(it, cfg.seeds_per_iter, base_seed)
        episodes = collect_fn(endpoint, gen_fn, seeds, cfg, sims, deadline, spec)
        stats = episode_stats(episodes)

        # Label: cold-start uses template Thoughts only when iter-1 is too invalid.
        cold = it == 0 and stats["valid_rate"] < cfg.cold_start_valid_threshold
        steps = filter_and_extract(episodes, keep_fraction=keep_frac,
                                   potential_fn=spec.potential_fn, feature_fn=spec.features_fn)
        log_fn(f"  [label] {len(steps)} winning steps -> STaR rationalize (cold_start={cold})")
        samples = synthesize(steps, gen_fn, cold_start=cold, system_prompt=spec.system_prompt)

        # Optional skill stream: self-instruct bash, exec-verified (same {messages}).
        n_skill = 0
        if skill_tasks_per_iter > 0:
            skill = collect_skill_samples(gen_fn, n_tasks=skill_tasks_per_iter, seed=base_seed + it)
            samples = samples + skill
            n_skill = len(skill)
        accumulated.extend(samples)

        # Improve: re-anchor SFT on the accumulated dataset.
        log_fn(f"  [improve] SFT on {len(accumulated)} accumulated samples (re-anchor from base)")
        dataset_path = dataset_fn(accumulated, os.path.join(out_dir, f"iter{it}_data"))
        adapter_path = sft_fn(base_model_path, dataset_path,
                              os.path.join(out_dir, f"iter{it}_sft"), cfg)

        rec = IterStats(it, stats["win_rate"], stats["valid_rate"], len(episodes),
                        len(samples), len(accumulated), adapter_path)
        history.append(rec)
        log_fn(
            f"[ReST iter {it}] DONE win_rate={rec.win_rate:.3f} valid_rate={rec.valid_rate:.3f} "
            f"episodes={rec.n_episodes} new={rec.n_new_samples} (skill={n_skill}) "
            f"total={rec.n_total_samples} took={time.monotonic() - t0:.0f}s"
        )

        kill.update(stats["win_rate"])
        if kill.should_pivot():
            log_fn(f"[ReST] kill-criterion hit after iter {it}: win-rate not improving; stopping.")
            break

    # Finalise: merge the last adapter into the base so OUTPUT_DIR is a full model.
    final_model_dir = merge_fn(base_model_path, adapter_path, out_dir, log_fn)

    return {
        "adapter_path": adapter_path,
        "final_model_dir": final_model_dir,
        "n_samples": len(accumulated),
        "history": [vars(h) for h in history],
    }


def _synthetic_format_samples() -> List[Dict[str, object]]:
    """A handful of trivial {messages} rows, used only to exercise the SFT+merge
    path when the (weak) smoke model solves too few real tasks."""
    qa = [
        ("What is 2+2?", "Thought:\nsimple sum.\n\nAction:\n4"),
        ("Name a primary color.", "Thought:\nred is primary.\n\nAction:\nred"),
        ("What is the capital of France?", "Thought:\nit is Paris.\n\nAction:\nParis"),
        ("How many days in a week?", "Thought:\nseven.\n\nAction:\n7"),
    ]
    sys = "You answer concisely in the Thought/Action format."
    return [
        {"messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": u},
            {"role": "assistant", "content": a},
        ]}
        for u, a in qa * 2
    ]


def run_skill_only(cfg: RestBucket, base_model_path: str, out_dir: str,
                   n_tasks: int = 8, log_fn: Callable[[str], None] = print) -> Optional[str]:
    """No env-server needed: validate generation + inner SFT + final merge using
    the self-instruct skill stream. A smoke convenience, not a tournament path."""
    gen = build_vllm_generate_fn(base_model_path, None, cfg)
    samples = collect_skill_samples(gen, n_tasks=n_tasks)
    log_fn(f"[skill-only] model solved {len(samples)}/{n_tasks} skill tasks")
    if len(samples) < 2:
        log_fn("[skill-only] too few solved; adding synthetic rows to exercise SFT+merge")
        samples = samples + _synthetic_format_samples()
    dataset_path = to_hf_dataset(samples, os.path.join(out_dir, "skill_data"))
    adapter = run_inner_sft(base_model_path, dataset_path,
                            os.path.join(out_dir, "skill_sft"), cfg)
    final = merge_and_save_final(base_model_path, adapter, out_dir, log_fn)
    log_fn(f"[skill-only] done: n_samples={len(samples)} final_model={final}")
    return final


def _resolve_inputs(argv=None) -> Dict[str, object]:
    """Resolve run inputs from CLI args, falling back to env vars, then to a
    training-request JSON (the shape text_trainer passes). CLI/env win."""
    import argparse

    p = argparse.ArgumentParser(description="ReST env trainer")
    p.add_argument("--model_path", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--task_id", default=None)
    p.add_argument("--env_server_urls", default=None)
    p.add_argument("--request_path", default=None)
    args, _ = p.parse_known_args(argv)

    req: Dict[str, object] = {}
    req_path = args.request_path or os.environ.get("REQUEST_PATH")
    if req_path and os.path.exists(req_path):
        try:
            with open(req_path) as f:
                req = json.load(f)
        except Exception:
            req = {}
    tr = req.get("train_request", req) if isinstance(req, dict) else {}

    model_path = args.model_path or os.environ.get("MODEL_PATH") or tr.get("model") or tr.get("model_path") or ""
    out_dir = args.output_dir or os.environ.get("OUTPUT_DIR") or tr.get("output_dir") or "/workspace/rest_output"
    task_id = args.task_id or os.environ.get("TASK_ID") or tr.get("task_id")
    raw_urls = args.env_server_urls or os.environ.get("ENVIRONMENT_SERVER_URLS", "")
    endpoints = [u.strip() for u in str(raw_urls).split(",") if u.strip()]
    return {"model_path": model_path, "out_dir": out_dir, "task_id": task_id, "endpoints": endpoints}


def main(argv=None) -> None:
    inp = _resolve_inputs(argv)
    base_model_path, out_dir = inp["model_path"], inp["out_dir"]
    endpoints, task_id = inp["endpoints"], inp["task_id"]
    if not base_model_path:
        raise RuntimeError("model_path is required (CLI/env/request)")
    cfg = get_rest_config(estimate_param_billions(base_model_path))

    # No-env-server validation path (smoke): generation + SFT + merge only.
    if os.environ.get("REST_SKILL_ONLY") == "1":
        run_skill_only(cfg, base_model_path, out_dir,
                       n_tasks=int(os.environ.get("REST_SKILL_TASKS") or "8") or 8)
        return

    if not endpoints:
        raise RuntimeError("env_server_urls required for game training (set ENVIRONMENT_SERVER_URLS)")
    spec = get_spec_by_task_id(int(task_id)) if task_id else get_spec(os.environ.get("GAME", "leduc_poker"))
    result = rest_loop(cfg, endpoints[0], base_model_path, out_dir, spec=spec,
                       skill_tasks_per_iter=int(os.environ.get("REST_SKILL_TASKS", "0")))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rest_history.json"), "w") as f:
        json.dump(result["history"], f, indent=2)
    print(f"[ReST] done: {result['n_samples']} samples, adapter={result['adapter_path']}, "
          f"final_model={result['final_model_dir']}")


# --------------------------------------------------------------------------- #
# Offline selftest: mock backends exercise the loop control flow (no GPU).
# --------------------------------------------------------------------------- #


def _selftest() -> None:
    os.environ["REST_USE_CFR"] = "0"  # heuristic Phi: keep the loop test fast
    from selfplay.rollout_collector import Turn
    from shaping import leduc_features

    def _win_episode(seed, sims):
        obs = "Your card: King\nCommunity card: King\nLegal Actions:\n2 -> Raise\nYour choice:"
        ep = Episode(game_id=LEDUC_GAME_ID, opponent_sims=sims, seed=seed)
        ep.turns = [Turn(obs, {"2": "Raise"}, obs, "Action:\n2", "2", 1.0, True, leduc_features(obs))]
        ep.terminal_reward = 1.0
        return ep

    # Budget allocation: iter 0 gets the heavier slice.
    b = RestBudget(total_seconds=100, first_iter_weight=2.0)
    d0 = b.iteration_deadline(0, 3) - b.start
    d1 = b.iteration_deadline(1, 3) - b.start
    assert abs(d0 - 50.0) < 1e-6 and abs(d1 - 75.0) < 1e-6, (d0, d1)

    # Kill-criterion fires on a flat win-rate window.
    k = KillCriterion(window=3, min_improvement=0.01)
    for wr in (0.2, 0.2, 0.2):
        k.update(wr)
    assert k.should_pivot()
    k2 = KillCriterion(window=3)
    for wr in (0.1, 0.3, 0.6):
        k2.update(wr)
    assert not k2.should_pivot()

    # Full loop with mock backends: assert re-anchor + accumulation + iteration count.
    sft_calls = []

    def mock_gen_factory(base, adapter, cfg):
        def gen(batches, n=1, temperature=1.0):
            return [["2"]] if "Action id only" in batches[0][-1]["content"] \
                else [["Thought:\npair, raise.\nAction:\n2"]]
        return gen

    def mock_sft(base, dataset_path, out, cfg):
        sft_calls.append({"base": base, "dataset": dataset_path})
        return f"{out}/adapter"

    def mock_collect(endpoint, gen, seeds, cfg, sims, deadline, spec):
        return [_win_episode(s, sims) for s in seeds[:3]]

    def mock_dataset(samples, out):
        return f"{out}::{len(samples)}"

    def mock_merge(base, adapter, out, log):
        return f"{out}/final" if adapter else None

    cfg = get_rest_config(7.0)
    cfg.num_iters = 2
    result = rest_loop(cfg, "http://x", "/models/base-7b", "/tmp/out",
                       gen_factory=mock_gen_factory, sft_fn=mock_sft,
                       collect_fn=mock_collect, dataset_fn=mock_dataset,
                       merge_fn=mock_merge, budget=RestBudget(1000), kill=KillCriterion(),
                       log_fn=lambda *_: None)
    assert len(result["history"]) == 2, result
    assert result["final_model_dir"] == "/tmp/out/final", result  # final merge ran
    # Re-anchor: every SFT call uses the base model path, never an adapter.
    assert all(c["base"] == "/models/base-7b" for c in sft_calls), sft_calls
    # Accumulation grows monotonically across iterations.
    totals = [h["n_total_samples"] for h in result["history"]]
    assert totals[1] > totals[0] >= 1, totals

    # Spec threading: the same loop runs for a different game without error.
    res_gin = rest_loop(cfg, "http://x", "/models/base-7b", "/tmp/out2",
                        spec=get_spec("gin_rummy"),
                        gen_factory=mock_gen_factory, sft_fn=mock_sft,
                        collect_fn=mock_collect, dataset_fn=mock_dataset,
                        merge_fn=mock_merge, budget=RestBudget(1000), kill=KillCriterion(),
                        log_fn=lambda *_: None)
    assert len(res_gin["history"]) == 2, res_gin
    print("rest_trainer selftest OK")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
