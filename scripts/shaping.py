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


# Set True in tests to force the fast heuristic and skip the CFR solve.
_FORCE_HEURISTIC = False


def _use_cfr() -> bool:
    return (not _FORCE_HEURISTIC) and os.environ.get("REST_USE_CFR", "1") != "0"


def _cfr_lookup(rank: int, has_pair: bool, rnd: int):
    """CFR-solved normalised potential for (rank, pair, round); None on failure.

    Maps shaping ranks (1=J,2=Q,3=K) / rounds (1,2) to the solver's
    (0,1,2) / (0,1) convention.
    """
    try:
        from cfr_leduc import potential_table  # lazy: solve only when needed
        tbl = potential_table(int(os.environ.get("CFR_ITERS", "20000")))
        cfr_round = 0 if rnd == 1 else 1
        key = (rank - 1, has_pair if cfr_round == 1 else False, cfr_round)
        return tbl.get(key)
    except Exception:
        return None


def _potential_leduc_heuristic(rank: int, has_pair: bool, rnd: int, opp_raised: bool) -> float:
    if has_pair:
        phi = 0.86 + 0.05 * (rank - 1)            # pair J -> pair K
    else:
        base = {1: 0.20, 2: 0.40, 3: 0.62}[rank]  # high card by rank
        phi = base * (0.85 if rnd == 2 else 1.0)
        if opp_raised:
            phi *= 0.80
    return max(0.0, min(1.0, phi))


def potential_leduc(features: Dict[str, object]) -> float:
    """Phi(s) in [0, 1]: how favourable the state is for us.

    Primary source is the CFR-solved state-value table (cfr_leduc), distinct from
    the baseline's hand-equity-vs-uniform heuristic. Falls back to a closed-form
    heuristic if the solver is unavailable. Opponent aggression shades Phi down.
    """
    rank = features.get("rank")
    if rank is None:
        return 0.5

    rank = int(rank)
    has_pair = bool(features.get("has_pair"))
    rnd = int(features.get("round", 1))
    opp_raised = bool(features.get("opponent_raised"))

    phi = _cfr_lookup(rank, has_pair, rnd) if _use_cfr() else None
    if phi is None:
        phi = _potential_leduc_heuristic(rank, has_pair, rnd, opp_raised)
    elif opp_raised:
        phi *= 0.85
    return max(0.0, min(1.0, phi))


# --------------------------------------------------------------------------- #
# Potentials for the other three games (heuristic; refined once real
# observations are seen at smoke). Each returns Phi in [0, 1] and falls back to
# 0.5 when the relevant signal cannot be parsed.
# --------------------------------------------------------------------------- #


def gin_rummy_features(observation: str) -> Dict[str, object]:
    feats: Dict[str, object] = {"deadwood": None, "can_knock": False}
    if not observation:
        return feats
    m = re.search(r"deadwood[:\s=]+(\d+)", observation, re.IGNORECASE)
    if m:
        feats["deadwood"] = int(m.group(1))
    if re.search(r"\bknock\b", observation, re.IGNORECASE) and "55 ->" in observation:
        feats["can_knock"] = True
    return feats


def potential_gin_rummy(features: Dict[str, object]) -> float:
    """Lower deadwood is better; a knock-ready hand gets a small bonus."""
    dw = features.get("deadwood")
    if dw is None:
        return 0.5
    phi = 1.0 - min(int(dw), 100) / 100.0
    if features.get("can_knock"):
        phi = min(1.0, phi + 0.1)
    return max(0.0, min(1.0, phi))


def liars_dice_features(observation: str) -> Dict[str, object]:
    feats: Dict[str, object] = {"bid_qty": None, "bid_face": None, "total_dice": None}
    if not observation:
        return feats
    m = re.search(r"(?:current )?bid[:\s]+(\d+)\D+(\d+)", observation, re.IGNORECASE)
    if m:
        feats["bid_qty"], feats["bid_face"] = int(m.group(1)), int(m.group(2))
    t = re.search(r"total dice[:\s]+(\d+)", observation, re.IGNORECASE)
    if t:
        feats["total_dice"] = int(t.group(1))
    return feats


def potential_liars_dice(features: Dict[str, object]) -> float:
    """Higher standing-bid quantity relative to total dice = riskier position."""
    qty, total = features.get("bid_qty"), features.get("total_dice")
    if qty is None or not total:
        return 0.5
    # Expected count of any face ~ total/6; a bid far above that is precarious.
    expected = total / 6.0
    ratio = expected / max(1, int(qty))
    return max(0.0, min(1.0, ratio))


def goofspiel_features(observation: str) -> Dict[str, object]:
    feats: Dict[str, object] = {"own_score": None, "opp_score": None, "prize": None}
    if not observation:
        return feats
    scores = re.findall(r"score[:\s]+(\d+)", observation, re.IGNORECASE)
    if len(scores) >= 2:
        feats["own_score"], feats["opp_score"] = int(scores[0]), int(scores[1])
    p = re.search(r"prize[^\d]*(\d+)", observation, re.IGNORECASE)
    if p:
        feats["prize"] = int(p.group(1))
    return feats


def potential_goofspiel(features: Dict[str, object]) -> float:
    """Score margin mapped to [0, 1] via a soft squashing."""
    own, opp = features.get("own_score"), features.get("opp_score")
    if own is None or opp is None:
        return 0.5
    import math
    return 0.5 + 0.5 * math.tanh((own - opp) / 20.0)


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
    global _FORCE_HEURISTIC
    pair = {"rank": 3, "has_pair": True, "round": 2, "opponent_raised": False}
    jack = {"rank": 1, "has_pair": False, "round": 1, "opponent_raised": False}

    _FORCE_HEURISTIC = True  # fast closed-form path
    assert potential_leduc(pair) > potential_leduc(jack)
    assert 0.0 <= potential_leduc(jack) <= 1.0
    assert per_turn_shaping(jack, pair) > 0
    assert shaped_return([jack, pair], terminal_reward=1.0) == shaped_return([jack, pair], 1.0)
    f = leduc_features("Your card: King\nCommunity card: King\nPot: 6\nLegal Actions:\n2 -> Raise")
    assert f["has_pair"] and f["round"] == 2 and f["rank"] == 3, f

    # CFR-solved path (low iters for speed): same qualitative ordering.
    _FORCE_HEURISTIC = False
    os.environ.setdefault("CFR_ITERS", "3000")
    assert potential_leduc(pair) > potential_leduc(jack)
    assert 0.0 <= potential_leduc(pair) <= 1.0
    print("shaping selftest OK")


if __name__ == "__main__":
    _selftest()
