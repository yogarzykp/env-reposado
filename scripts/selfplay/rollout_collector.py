"""Grow phase: self-play rollout collection for Leduc poker.

The collector drives whole episodes against the validator's env-server. At every
decision point it asks the *policy under training* (injected as ``generate_fn``)
for a ``Thought:/Action:`` response, parses the action, and steps the env. It
records the full trajectory plus the terminal outcome so the downstream
hard-filter can keep only winning play (rejection sampling).

Design notes (env-reposado, distinct from the SFT->GRPO baseline):
  - Output contract is ``Thought:\\n...\\nAction:\\n<id>`` (not action-only). This
    matches the validator's reasoning-capable parser and the STaR rationalization
    used downstream.
  - The MCTS opponent strength (``mcts_max_simulations``) is a *curriculum* knob
    passed per reset, so iteration 1 can face a weak opponent and find wins to
    bootstrap from (cold-start mitigation).
  - ``generate_fn`` is dependency-injected so this module is unit-testable without
    vLLM / a GPU. The vLLM backing is wired in Fase 2 (rest_trainer).

The HTTP protocol (POST /reset, POST /step with the ``result`` envelope) mirrors
the contract used by ``env_function/leduc_poker_environment_function.py``; it is
re-implemented here as a thin client so this package imports cleanly without
pulling the TRL/vLLM stack.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

# A generator maps a batch of chat prompts to, per prompt, a list of `n`
# completions. Each chat prompt is a list of {"role", "content"} messages; the
# backend applies the model's chat template (wired in rest_trainer, Fase 2).
#   generate_fn(message_batches, n=1, temperature=1.0, max_new_tokens=512)
#       -> List[List[str]]
Messages = List[Dict[str, str]]
GenerateFn = Callable[..., List[List[str]]]

# Leduc poker lives in this validator task-id band (see env_function contract).
LEDUC_TASK_ID_RANGE = (200000000, 299999999)

DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_MAX_TURNS = 12

# Our own Thought/Action system prompt. Kept factual and short; the salient
# difference from the baseline leduc prompt is that we *require* reasoning before
# the action rather than forbidding it.
LEDUC_SYSTEM_PROMPT = (
    "You are playing Leduc Hold'em poker against one opponent.\n"
    "Deck: two suits each of Jack, Queen, King (six cards). Ranking K > Q > J; a "
    "card that matches the community card is a pair and beats any high card.\n"
    "There are two betting rounds; the community card is revealed before round two.\n"
    "\n"
    "On every turn you receive the current state and a list of legal actions, each "
    "written as `<id> -> <name>`.\n"
    "Respond in exactly this format and nothing else:\n"
    "Thought:\n"
    "<one or two sentences of reasoning about your hand and the action>\n"
    "\n"
    "Action:\n"
    "<the integer id of one legal action>\n"
)


@dataclass
class Turn:
    """One decision point inside an episode."""

    observation: str
    legal_actions: Dict[str, str]
    user_prompt: str
    model_output: str
    action_id: str
    step_reward: float
    valid: bool
    state_features: Dict[str, object] = field(default_factory=dict)


@dataclass
class Episode:
    """A full self-play game plus its outcome."""

    game_id: int
    opponent_sims: int
    seed: Optional[int]
    turns: List[Turn] = field(default_factory=list)
    terminal_reward: float = 0.0
    done: bool = False
    num_invalid: int = 0

    @property
    def won(self) -> bool:
        return self.terminal_reward > 0.0

    @property
    def num_turns(self) -> int:
        return len(self.turns)


# --------------------------------------------------------------------------- #
# Parsing helpers (contract layer, deliberately small and dependency-free)
# --------------------------------------------------------------------------- #

_LEGAL_BLOCK_RE = re.compile(
    r"Legal Actions:\s*\n(.*?)(?:\n\nYour choice|\nYour choice|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_INT_RE = re.compile(r"-?\d+")


def extract_legal_actions(observation: str) -> Dict[str, str]:
    """Parse the ``<id> -> <name>`` legal-action block from an observation."""
    if not observation:
        return {}
    match = _LEGAL_BLOCK_RE.search(observation)
    if not match:
        return {}
    mapping: Dict[str, str] = {}
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "->" in line:
            left, right = line.split("->", 1)
            action_id, label = left.strip(), right.strip()
        else:
            action_id = label = line
        if _INT_RE.fullmatch(action_id):
            mapping[action_id] = label
    return mapping


def parse_thought_action(completion: str, legal_actions: Dict[str, str]) -> str:
    """Return the legal action id parsed from a Thought/Action completion.

    Returns "" when no legal action can be recovered (an invalid response).
    """
    if not legal_actions or not completion:
        return ""
    text = completion
    # Keep only the post-"Action:" segment when present (the STaR output shape).
    if "Action:" in text:
        text = text.split("Action:")[-1]
    for num in _INT_RE.findall(text):
        if num in legal_actions:
            return num
    # Fall back to matching an action *name* if the model wrote a word.
    low = text.strip().lower()
    for action_id, label in legal_actions.items():
        if low == label.strip().lower():
            return action_id
    for keyword in ("fold", "check", "call", "raise", "bet"):
        if keyword in low:
            for action_id, label in legal_actions.items():
                if keyword in label.strip().lower():
                    return action_id
    return ""


def build_user_prompt(observation: str) -> str:
    """Wrap a raw env observation as the user turn shown to the policy."""
    return observation if observation else ""


def build_messages(observation: str) -> Messages:
    """Chat prompt for one decision point: system rules + the current state."""
    return [
        {"role": "system", "content": LEDUC_SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(observation)},
    ]


# --------------------------------------------------------------------------- #
# Thin env-server client (POST /reset, POST /step)
# --------------------------------------------------------------------------- #


def _server_urls() -> List[str]:
    raw = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
    return [u.strip() for u in raw.split(",") if u.strip()]


def reset_episode(
    endpoint: str,
    game_id: int,
    opponent_sims: int,
    seed: Optional[int] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
):
    """Reset the env-server. Returns ``(episode_id, observation)``.

    ``opponent_sims`` is the per-reset MCTS strength used for curriculum.
    """
    import requests  # lazy: keep module import dependency-free

    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    payload = {
        "task_id": game_id,
        "seed": seed,
        "opponent": "mcts",
        "mcts_max_simulations": int(opponent_sims),
        "mcts_num_rollouts": 1,
    }
    res = requests.post(f"{endpoint}/reset", json=payload, timeout=timeout)
    res.raise_for_status()
    result = res.json()["result"]
    return result.get("episode_id", ""), result.get("observation", "")


def step_episode(
    endpoint: str,
    episode_id: str,
    action_id: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
):
    """Send one action. Returns ``(observation, reward, done)``."""
    import requests  # lazy

    payload = {"action": action_id, "episode_id": episode_id}
    res = requests.post(f"{endpoint}/step", json=payload, timeout=timeout)
    res.raise_for_status()
    block = res.json()["result"]
    reward = float(block.get("reward", 0.0) or 0.0)
    return block.get("observation", ""), reward, bool(block.get("done", False))


# --------------------------------------------------------------------------- #
# Episode rollout
# --------------------------------------------------------------------------- #


def play_episode(
    endpoint: str,
    game_id: int,
    opponent_sims: int,
    generate_fn: GenerateFn,
    temperature: float = 1.0,
    max_turns: int = DEFAULT_MAX_TURNS,
    seed: Optional[int] = None,
    feature_fn: Optional[Callable[[str], Dict[str, object]]] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> Episode:
    """Play a single self-play episode and return the recorded trajectory.

    On an invalid model action the turn is recorded as invalid and a *random*
    legal action is injected so the episode can continue (better cold-start data
    yield). The fallback is deliberately random, not expert, to keep the
    format-priming red line: we never hand-code strategy into the data. Invalid
    turns carry ``valid=False`` and are excluded from the SFT set by the filter.
    """
    episode_id, observation = reset_episode(endpoint, game_id, opponent_sims, seed, timeout)
    episode = Episode(game_id=game_id, opponent_sims=opponent_sims, seed=seed)

    for _ in range(max_turns):
        legal = extract_legal_actions(observation)
        if not legal:
            break
        user_prompt = build_user_prompt(observation)
        completion = generate_fn([build_messages(observation)], n=1, temperature=temperature)[0][0]
        action_id = parse_thought_action(completion, legal)
        valid = bool(action_id)
        features = feature_fn(observation) if feature_fn else {}

        if valid:
            action_to_send = action_id
        else:
            episode.num_invalid += 1
            action_to_send = _random_fallback_action(legal)

        next_obs, reward, done = step_episode(endpoint, episode_id, action_to_send, timeout)
        # Record the model's parsed action_id (empty when invalid), not the
        # fallback: invalid turns must not teach the injected action.
        episode.turns.append(
            Turn(observation, legal, user_prompt, completion, action_id, reward, valid, features)
        )
        episode.terminal_reward = reward
        observation = next_obs
        if done:
            episode.done = True
            break

    return episode


def _random_fallback_action(legal_actions: Dict[str, str]) -> str:
    """A uniformly random legal action id (generic, never expert strategy)."""
    return random.choice(list(legal_actions.keys()))


def collect_episodes(
    endpoint: str,
    generate_fn: GenerateFn,
    seeds: Sequence[int],
    n_per_seed: int = 4,
    opponent_sims: int = 10,
    temperature: float = 1.0,
    max_turns: int = DEFAULT_MAX_TURNS,
    game_id: Optional[int] = None,
    feature_fn: Optional[Callable[[str], Dict[str, object]]] = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> List[Episode]:
    """Best-of-N self-play: ``n_per_seed`` stochastic episodes per seed.

    Episodes diverge through temperature sampling; the downstream filter keeps the
    best by outcome. ``opponent_sims`` is the curriculum strength for this batch.
    """
    if game_id is None:
        game_id = LEDUC_TASK_ID_RANGE[0]
    episodes: List[Episode] = []
    for seed in seeds:
        for _ in range(max(1, n_per_seed)):
            episodes.append(
                play_episode(
                    endpoint,
                    game_id,
                    opponent_sims,
                    generate_fn,
                    temperature=temperature,
                    max_turns=max_turns,
                    seed=seed,
                    feature_fn=feature_fn,
                    timeout=timeout,
                )
            )
    return episodes


# --------------------------------------------------------------------------- #
# Offline self-test (no network, no model): exercises parsing + dataclasses.
# --------------------------------------------------------------------------- #


def _selftest() -> None:
    obs = (
        "Your card: King\n"
        "Community card: None\n"
        "Pot: 2\n"
        "Legal Actions:\n"
        "0 -> Fold\n"
        "1 -> Call\n"
        "2 -> Raise\n"
        "Your choice:"
    )
    legal = extract_legal_actions(obs)
    assert legal == {"0": "Fold", "1": "Call", "2": "Raise"}, legal
    assert parse_thought_action("Thought:\nKing is strong.\n\nAction:\n2", legal) == "2"
    assert parse_thought_action("Action:\n9", legal) == ""  # 9 not legal
    assert parse_thought_action("i will raise", legal) == "2"

    # A scripted generator + patched client lets us run a fake episode. We patch
    # this module's own globals (not a re-import) to stay correct under `-m`.
    # Turn 1 is invalid ("9" is not legal) -> random fallback, episode continues;
    # turn 2 is a valid winning move.
    scripted = iter(["Thought:\nconfused.\nAction:\n9", "Thought:\ncall.\nAction:\n1"])
    steps = iter([("Pot: 4\nLegal Actions:\n0 -> Fold\n1 -> Call\nYour choice:", 0.0, False),
                  ("done", 1.0, True)])
    g = globals()
    g["reset_episode"] = lambda *a, **k: ("ep1", obs)
    g["step_episode"] = lambda *a, **k: next(steps)
    ep = play_episode("http://x", LEDUC_TASK_ID_RANGE[0], 5,
                      lambda prompts, n=1, temperature=1.0: [[next(scripted)]])
    assert ep.won and ep.num_turns == 2 and ep.num_invalid == 1, ep
    assert ep.turns[0].valid is False and ep.turns[0].action_id == "", ep.turns[0]
    assert ep.turns[1].valid is True and ep.turns[1].action_id == "1", ep.turns[1]
    print("rollout_collector selftest OK")


if __name__ == "__main__":
    _selftest()
