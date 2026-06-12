"""Potential-based reward shaping for env-reposado.

We shape the sparse env reward with a per-game potential ``Phi(s)`` and the
classic policy-invariant form

    F(s, s') = gamma * Phi(s') - Phi(s)

Adding ``F`` to the reward provably leaves the optimal policy unchanged
(Ng, Harada & Russell, 1999, "Policy invariance under reward transformations").
That guarantee is the whole point: we densify the win/loss signal for *ranking
and credit assignment* inside the ReST hard-filter without corrupting the true
objective (win the game), which stays the filter gate.

This is intentionally NOT the additive event-bonus shaping used by the baseline
(fixed +set/+run/+knock or hand-equity-vs-uniform). The shaping here is a single
coherent potential per game and enters the pipeline only as a ranking / pruning
signal.

For Leduc the potential is a small CFR-flavoured state-value table; replacing it
with values from an actual CFR solve is deferred to Fase 3.
"""

from __future__ import annotations

import os
import re
from typing import Callable, Dict, List, Optional

# Discount for the shaping term. Standalone from any baseline default.
DEFAULT_GAMMA = float(os.environ.get("REST_GAMMA") or "0.97")

_RANK = {"j": 1, "jack": 1, "q": 2, "queen": 2, "k": 3, "king": 3}


def leduc_features(observation: str) -> Dict[str, object]:
    """Lightweight, self-contained state parse for the Leduc potential.

    Returns rank (1..3), pair flag, round (1/2) and an opponent-aggression flag.
    Kept separate from the env_function parser so this module imports cleanly.
    """
    feats: Dict[str, object] = {
        "rank": None,
        "has_pair": False,
        "round": 1,
        "opponent_raised": False,
    }
    if not observation:
        return feats

    priv = re.search(r"(?:Your|Private|Hole)\s*(?:card|hand)\s*[:\s]+\s*(\w+)",
                     observation, re.IGNORECASE)
    board = re.search(r"(?:Community|Public|Board|Flop)\s*(?:card)?\s*[:\s]+\s*(\w+)",
                      observation, re.IGNORECASE)

    priv_rank = _RANK.get(priv.group(1).lower()) if priv else None
    feats["rank"] = priv_rank

    board_rank = None
    if board:
        token = board.group(1).strip().lower()
        if token not in ("none", "n/a", "not", "no"):
            board_rank = _RANK.get(token)
            feats["round"] = 2
    if priv_rank is not None and board_rank is not None and priv_rank == board_rank:
        feats["has_pair"] = True

    if re.search(r"(?:opponent|player\s*\d)\s+(?:raised|raises|bet|bets)",
                 observation, re.IGNORECASE):
        feats["opponent_raised"] = True
    return feats


def potential_leduc(features: Dict[str, object]) -> float:
    """Phi(s) in [0, 1]: our estimate of how favourable the state is for us.

    Construction (CFR-flavoured table, distinct from the baseline heuristic):
      - a pair is the nut hand -> near-maximal potential, scaled by rank;
      - otherwise a high-card value that is lower in round 2 (more cards live
        against us) and discounted when the opponent has shown aggression.
    """
    rank = features.get("rank")
    if rank is None:
        return 0.5

    rank = int(rank)  # 1=J, 2=Q, 3=K
    has_pair = bool(features.get("has_pair"))
    rnd = int(features.get("round", 1))
    opp_raised = bool(features.get("opponent_raised"))

    if has_pair:
        # 0.86 (pair of J) -> 0.96 (pair of K).
        phi = 0.86 + 0.05 * (rank - 1)
    else:
        # High-card baseline by rank, softened in round 2.
        base = {1: 0.20, 2: 0.40, 3: 0.62}[rank]
        phi = base * (0.85 if rnd == 2 else 1.0)
        if opp_raised:
            # Aggression signals strength; shade our potential down.
            phi *= 0.80
    return max(0.0, min(1.0, phi))


# Potential of a terminal (absorbing) state is 0 by convention.
TERMINAL_POTENTIAL = 0.0


def per_turn_shaping(
    prev_features: Dict[str, object],
    next_features: Optional[Dict[str, object]],
    gamma: float = DEFAULT_GAMMA,
    potential_fn: Callable[[Dict[str, object]], float] = potential_leduc,
) -> float:
    """F = gamma * Phi(s') - Phi(s). ``next_features=None`` => terminal."""
    phi_prev = potential_fn(prev_features)
    phi_next = TERMINAL_POTENTIAL if next_features is None else potential_fn(next_features)
    return gamma * phi_next - phi_prev


def shaping_series(
    feature_seq: List[Dict[str, object]],
    gamma: float = DEFAULT_GAMMA,
    potential_fn: Callable[[Dict[str, object]], float] = potential_leduc,
) -> List[float]:
    """Per-turn F values for a sequence of states (last transition is terminal)."""
    out: List[float] = []
    n = len(feature_seq)
    for t in range(n):
        nxt = feature_seq[t + 1] if t + 1 < n else None
        out.append(per_turn_shaping(feature_seq[t], nxt, gamma, potential_fn))
    return out


def shaped_return(
    feature_seq: List[Dict[str, object]],
    terminal_reward: float,
    gamma: float = DEFAULT_GAMMA,
    potential_fn: Callable[[Dict[str, object]], float] = potential_leduc,
) -> float:
    """Terminal reward plus discounted shaping; a finer-grained ranking score."""
    fs = shaping_series(feature_seq, gamma, potential_fn)
    return terminal_reward + sum((gamma ** t) * f for t, f in enumerate(fs))


def _selftest() -> None:
    pair = {"rank": 3, "has_pair": True, "round": 2, "opponent_raised": False}
    jack = {"rank": 1, "has_pair": False, "round": 1, "opponent_raised": False}
    assert potential_leduc(pair) > potential_leduc(jack)
    assert 0.0 <= potential_leduc(jack) <= 1.0
    # Improving from J-high to a King pair gives a positive shaping transition.
    assert per_turn_shaping(jack, pair) > 0
    seq = [jack, pair]
    assert shaped_return(seq, terminal_reward=1.0) > 1.0 - 1.0  # sanity: finite
    f = leduc_features("Your card: King\nCommunity card: King\nPot: 6\nLegal Actions:\n2 -> Raise")
    assert f["has_pair"] and f["round"] == 2 and f["rank"] == 3, f
    print("shaping selftest OK")


if __name__ == "__main__":
    _selftest()
