"""Hard-filter (rejection sampling) for self-play episodes.

This is the "Improve gate" of one ReST iteration. Three stages, in order:

  1. GATE   - keep only episodes that actually won (raw terminal outcome). This
              keeps the supervised target aligned with the validator's true
              objective; shaping never decides the gate.
  2. RANK    - among winners, rank by shaped return (terminal + potential-based
              shaping) and keep the top ``keep_fraction`` (a relative top-k%,
              robust to cold start: there is always a survivor).
  3. PRUNE   - inside each kept winning episode, drop individual turns whose
              per-turn shaping F is strongly negative (a clearly bad move that
              happened inside a won game), so we do not teach those moves. The
              decisive final turn is always retained.

The survivors are flattened into ``TrainingStep`` records for STaR
rationalization (cot_synthesizer).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List

from shaping import (
    DEFAULT_GAMMA,
    leduc_features,
    potential_leduc,
    shaped_return,
    shaping_series,
)

# Default: drop a turn only on a clearly bad potential drop. Conservative so we
# do not discard bluffs that worked.
DEFAULT_PRUNE_F = float(os.environ.get("REST_PRUNE_F") or "-0.35")


@dataclass
class TrainingStep:
    """A single (state -> winning action) example bound for rationalization."""

    observation: str
    user_prompt: str
    action_id: str
    action_label: str
    legal_actions: Dict[str, str]
    features: Dict[str, object] = field(default_factory=dict)
    source_game_id: int = 0


def _episode_feature_seq(episode, feature_fn) -> List[Dict[str, object]]:
    """Per-turn features for an episode, reusing recorded features when present."""
    seq: List[Dict[str, object]] = []
    for turn in episode.turns:
        if turn.state_features:
            seq.append(turn.state_features)
        else:
            seq.append(feature_fn(turn.observation))
    return seq


def gate_won(episodes: List) -> List:
    """Stage 1: raw-outcome gate."""
    return [ep for ep in episodes if ep.won]


def rank_and_select(
    episodes: List,
    keep_fraction: float,
    gamma: float = DEFAULT_GAMMA,
    potential_fn: Callable[[Dict[str, object]], float] = potential_leduc,
    feature_fn: Callable[[str], Dict[str, object]] = leduc_features,
) -> List:
    """Stage 2: rank winners by shaped return, keep relative top-k%."""
    if not episodes:
        return []
    scored = []
    for ep in episodes:
        seq = _episode_feature_seq(ep, feature_fn)
        score = shaped_return(seq, ep.terminal_reward, gamma, potential_fn)
        scored.append((score, ep))
    scored.sort(key=lambda x: x[0], reverse=True)
    keep_n = max(1, math.ceil(len(scored) * max(0.0, min(1.0, keep_fraction))))
    return [ep for _, ep in scored[:keep_n]]


def prune_turns(
    episode,
    gamma: float = DEFAULT_GAMMA,
    prune_f: float = DEFAULT_PRUNE_F,
    potential_fn: Callable[[Dict[str, object]], float] = potential_leduc,
    feature_fn: Callable[[str], Dict[str, object]] = leduc_features,
) -> List:
    """Stage 3: return the kept (valid) turns of one winning episode."""
    seq = _episode_feature_seq(episode, feature_fn)
    fseries = shaping_series(seq, gamma, potential_fn)
    last_valid_idx = max((i for i, t in enumerate(episode.turns) if t.valid), default=-1)
    kept = []
    for i, turn in enumerate(episode.turns):
        if not turn.valid:
            continue
        if i == last_valid_idx:  # always keep the decisive move
            kept.append(turn)
            continue
        if fseries[i] >= prune_f:
            kept.append(turn)
    return kept


def filter_and_extract(
    episodes: List,
    keep_fraction: float,
    gamma: float = DEFAULT_GAMMA,
    prune_f: float = DEFAULT_PRUNE_F,
    potential_fn: Callable[[Dict[str, object]], float] = potential_leduc,
    feature_fn: Callable[[str], Dict[str, object]] = leduc_features,
) -> List[TrainingStep]:
    """Run gate -> rank -> prune and flatten survivors to TrainingStep records."""
    winners = gate_won(episodes)
    selected = rank_and_select(winners, keep_fraction, gamma, potential_fn, feature_fn)
    steps: List[TrainingStep] = []
    for ep in selected:
        for turn in prune_turns(ep, gamma, prune_f, potential_fn, feature_fn):
            steps.append(
                TrainingStep(
                    observation=turn.observation,
                    user_prompt=turn.user_prompt,
                    action_id=turn.action_id,
                    action_label=turn.legal_actions.get(turn.action_id, turn.action_id),
                    legal_actions=turn.legal_actions,
                    features=turn.state_features or feature_fn(turn.observation),
                    source_game_id=ep.game_id,
                )
            )
    return steps


def _selftest() -> None:
    os.environ["REST_USE_CFR"] = "0"  # heuristic Phi: keep the test fast
    from selfplay.rollout_collector import Episode, Turn

    def mk_turn(obs, aid, reward, valid=True):
        legal = {"0": "Fold", "1": "Call", "2": "Raise"}
        return Turn(obs, legal, obs, f"Action:\n{aid}", aid, reward, valid, leduc_features(obs))

    win = Episode(game_id=200000000, opponent_sims=5, seed=1)
    win.turns = [
        mk_turn("Your card: King\nLegal Actions:\n2 -> Raise", "2", 0.0),
        mk_turn("Your card: King\nCommunity card: King\nLegal Actions:\n2 -> Raise", "2", 1.0),
    ]
    win.terminal_reward = 1.0
    lose = Episode(game_id=200000000, opponent_sims=5, seed=2)
    lose.turns = [mk_turn("Your card: Jack\nLegal Actions:\n0 -> Fold", "0", -1.0)]
    lose.terminal_reward = -1.0

    assert gate_won([win, lose]) == [win]
    steps = filter_and_extract([win, lose], keep_fraction=1.0)
    assert steps and all(s.action_id in s.legal_actions for s in steps), steps
    assert steps[-1].action_label == "Raise"
    print(f"trajectory_filter selftest OK ({len(steps)} steps kept)")


if __name__ == "__main__":
    _selftest()
