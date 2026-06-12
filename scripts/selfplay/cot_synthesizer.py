"""STaR rationalization: attach a Thought to each kept (state -> winning action).

Self-play yields the winning *action* but no reasoning. To produce the
``Thought:/Action:`` data the policy is trained on, we follow STaR
(Zelikman et al. 2022, arXiv 2203.14465) *rationalization*: the action is already
known to be good, so we hand it to the model as a hint and ask it to write the
reasoning that justifies arriving at it.

Two guard rails make this distinct and trustworthy:

  - SELF-CONTAINED: the rationalizer is the *same policy under training*
    (injected ``generate_fn``). We never call an external teacher model, so the
    data stays on-policy and does not drift toward any external distribution.
  - CONSISTENCY FILTER: after a Thought is written we re-ask the model for the
    action *from the Thought alone* (action hidden). Only Thoughts that lead back
    to the same action survive. This drops hallucinated rationales and is a step
    the baseline pipeline does not have.

During cold start (iteration 1, model can barely play) we fall back to a minimal
feature-templated Thought purely to bootstrap valid format; strategic
rationalization takes over once the policy starts winning.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from selfplay.rollout_collector import (
    LEDUC_SYSTEM_PROMPT,
    GenerateFn,
    Messages,
    parse_thought_action,
)
from selfplay.trajectory_filter import TrainingStep

_RANK_DESC = {1: "a weak Jack high", 2: "a middling Queen high", 3: "a strong King high"}


def _describe_hand(features: Dict[str, object]) -> str:
    if features.get("has_pair"):
        return "a pair, the strongest hand"
    rank = features.get("rank")
    return _RANK_DESC.get(int(rank), "an unclear hand") if rank else "an unclear hand"


def template_thought(step: TrainingStep) -> str:
    """Minimal, model-free Thought for cold-start bootstrapping (format only)."""
    return f"I hold {_describe_hand(step.features)}, so I play {step.action_label}."


def build_rationalize_prompt(step: TrainingStep, system_prompt: str = LEDUC_SYSTEM_PROMPT) -> Messages:
    """STaR hint prompt: reveal the good action, ask only for the reasoning."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"{step.observation}\n\n"
                f"The strong move here is action {step.action_id} ({step.action_label}). "
                f"In one or two sentences, explain the reasoning that leads to it. "
                f"Write only the reasoning, no action id."
            ),
        },
    ]


def build_verify_prompt(step: TrainingStep, thought: str,
                        system_prompt: str = LEDUC_SYSTEM_PROMPT) -> Messages:
    """Consistency prompt: recover the action from the Thought alone."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"{step.observation}\n\n"
                f"Reasoning: {thought}\n"
                f"Given this reasoning, output the Action id only."
            ),
        },
    ]


def clean_thought(raw: str) -> str:
    """Strip any leaked format tokens and keep a tidy one-liner."""
    text = raw or ""
    if "Action:" in text:
        text = text.split("Action:")[0]
    text = text.replace("Thought:", " ").strip()
    # Collapse to a single compact sentence-ish line.
    text = " ".join(text.split())
    return text[:400]


def synthesize_thought(
    step: TrainingStep,
    generate_fn: GenerateFn,
    temperature: float = 0.7,
    cold_start: bool = False,
    system_prompt: str = LEDUC_SYSTEM_PROMPT,
) -> str:
    if cold_start:
        return template_thought(step)
    raw = generate_fn([build_rationalize_prompt(step, system_prompt)],
                      n=1, temperature=temperature)[0][0]
    thought = clean_thought(raw)
    return thought or template_thought(step)


def consistency_ok(step: TrainingStep, thought: str, generate_fn: GenerateFn,
                   system_prompt: str = LEDUC_SYSTEM_PROMPT) -> bool:
    raw = generate_fn([build_verify_prompt(step, thought, system_prompt)],
                      n=1, temperature=0.0)[0][0]
    return parse_thought_action(raw, step.legal_actions) == step.action_id


def build_sft_sample(step: TrainingStep, thought: str,
                     system_prompt: str = LEDUC_SYSTEM_PROMPT) -> Dict[str, object]:
    """One {messages: [...]} record with a Thought/Action assistant turn."""
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": step.user_prompt or step.observation},
            {"role": "assistant", "content": f"Thought:\n{thought}\n\nAction:\n{step.action_id}"},
        ]
    }


def synthesize(
    steps: List[TrainingStep],
    generate_fn: GenerateFn,
    temperature: float = 0.7,
    cold_start: bool = False,
    consistency: bool = True,
    system_prompt: str = LEDUC_SYSTEM_PROMPT,
) -> List[Dict[str, object]]:
    """Rationalize every kept step into an SFT sample, dropping inconsistent ones."""
    samples: List[Dict[str, object]] = []
    for step in steps:
        thought = synthesize_thought(step, generate_fn, temperature, cold_start, system_prompt)
        if consistency and not cold_start and not consistency_ok(
                step, thought, generate_fn, system_prompt):
            continue
        samples.append(build_sft_sample(step, thought, system_prompt))
    return samples


def _selftest() -> None:
    step = TrainingStep(
        observation="Your card: King\nCommunity card: King\nLegal Actions:\n2 -> Raise\nYour choice:",
        user_prompt="Your card: King\nCommunity card: King\nLegal Actions:\n2 -> Raise\nYour choice:",
        action_id="2",
        action_label="Raise",
        legal_actions={"0": "Fold", "1": "Call", "2": "Raise"},
        features={"rank": 3, "has_pair": True, "round": 2, "opponent_raised": False},
        source_game_id=200000000,
    )

    def _user_text(prompts):
        return prompts[0][-1]["content"]

    # Generator: rationalize -> reasoning text; verify -> the correct action id.
    def gen(prompts, n=1, temperature=1.0):
        if "output the Action id only" in _user_text(prompts):
            return [["2"]]
        return [["Thought:\nA pair of Kings is the nuts, so raise for value.\nAction:\n2"]]

    out = synthesize([step], gen, cold_start=False, consistency=True)
    assert len(out) == 1, out
    msg = out[0]["messages"]
    assert msg[2]["content"].startswith("Thought:") and "Action:\n2" in msg[2]["content"], msg
    assert "Action:" not in msg[2]["content"].split("Action:\n2")[0].replace("Action:\n", "X")

    # Cold-start path is model-free and always emits a template Thought.
    cold = synthesize([step], gen, cold_start=True)
    assert "pair" in cold[0]["messages"][2]["content"].lower()

    # Inconsistent verifier -> sample dropped.
    def gen_bad(prompts, n=1, temperature=1.0):
        if "output the Action id only" in _user_text(prompts):
            return [["0"]]
        return [["raise for value"]]

    assert synthesize([step], gen_bad, consistency=True) == []
    print("cot_synthesizer selftest OK")


if __name__ == "__main__":
    _selftest()
