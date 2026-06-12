"""Counterfactual Regret Minimization solver for Leduc Hold'em.

This produces an approximate Nash strategy and, from it, a per-state value table
that the shaping module uses as the Leduc potential ``Phi`` (Crux B decision:
leduc shaping basis is CFR, not the baseline's hand-equity-vs-uniform heuristic).

Algorithm: chance-sampled CFR (Lanctot et al. 2009) over the standard Leduc tree.
Reference: Zinkevich et al. 2007, "Regret Minimization in Games with Incomplete
Information" (NeurIPS).

Game (standard Leduc): 6 cards = ranks {0:J, 1:Q, 2:K} x 2 suits; ante 1 each;
round 0 bet size 2, round 1 (one community card) bet size 4; at most two raises
per round; a card matching the board is a pair and beats any high card.

Everything here is pure Python and offline-testable (no GPU, no env-server).
"""

from __future__ import annotations

import os
import random
from functools import lru_cache
from typing import Dict, List, Tuple

RANKS = (0, 1, 2)            # J, Q, K
DECK = [0, 0, 1, 1, 2, 2]
BET_SIZE = (2, 4)           # round 0, round 1
MAX_RAISES = 2


class _Node:
    __slots__ = ("actions", "regret_sum", "strategy_sum")

    def __init__(self, actions: List[str]):
        self.actions = actions
        self.regret_sum = {a: 0.0 for a in actions}
        self.strategy_sum = {a: 0.0 for a in actions}

    def strategy(self) -> Dict[str, float]:
        pos = {a: max(self.regret_sum[a], 0.0) for a in self.actions}
        total = sum(pos.values())
        if total > 0:
            return {a: pos[a] / total for a in self.actions}
        return {a: 1.0 / len(self.actions) for a in self.actions}

    def average_strategy(self) -> Dict[str, float]:
        total = sum(self.strategy_sum.values())
        if total > 0:
            return {a: self.strategy_sum[a] / total for a in self.actions}
        return {a: 1.0 / len(self.actions) for a in self.actions}


def _legal_actions(rhist: str, to_call: int) -> List[str]:
    raises = rhist.count("r")
    if to_call > 0:
        acts = ["f", "c"]
    else:
        acts = ["c"]
    if raises < MAX_RAISES:
        acts.append("r")
    return acts


def _round_over(rhist: str) -> bool:
    # A call/check that is not the opening action closes the round.
    return len(rhist) >= 2 and rhist[-1] == "c"


def _winner(c0: int, c1: int, board: int) -> int:
    """Return 0 if P0 wins, 1 if P1 wins, -1 on a tie."""
    p0_pair, p1_pair = c0 == board, c1 == board
    if p0_pair and not p1_pair:
        return 0
    if p1_pair and not p0_pair:
        return 1
    if c0 > c1:
        return 0
    if c1 > c0:
        return 1
    return -1


def _infoset_key(card: int, board, round_idx: int, rhist: str) -> str:
    return f"{card}:{board if board is not None else '-'}:{round_idx}:{rhist}"


def _cfr(nodes, cards, round_idx, rhist, contrib, reach0, reach1):
    c0, c1, board = cards
    player = len(rhist) % 2  # P0 opens each round

    # Terminal: fold.
    if rhist.endswith("f"):
        folder = (len(rhist) - 1) % 2
        return contrib[1] if folder == 1 else -contrib[0]

    # Round resolution.
    if _round_over(rhist):
        if round_idx == 0:
            return _cfr(nodes, cards, 1, "", contrib, reach0, reach1)
        win = _winner(c0, c1, board)
        if win == 0:
            return contrib[1]
        if win == 1:
            return -contrib[0]
        return 0.0

    to_call = max(contrib) - contrib[player]
    actions = _legal_actions(rhist, to_call)
    own_card = c0 if player == 0 else c1
    key = _infoset_key(own_card, board if round_idx == 1 else None, round_idx, rhist)
    node = nodes.get(key)
    if node is None:
        node = nodes[key] = _Node(actions)

    strat = node.strategy()
    util = {}
    node_v0 = 0.0
    for a in actions:
        new_contrib = list(contrib)
        if a == "c":
            new_contrib[player] += to_call
        elif a == "r":
            new_contrib[player] += to_call + BET_SIZE[round_idx]
        if player == 0:
            v0 = _cfr(nodes, cards, round_idx, rhist + a, new_contrib,
                      reach0 * strat[a], reach1)
        else:
            v0 = _cfr(nodes, cards, round_idx, rhist + a, new_contrib,
                      reach0, reach1 * strat[a])
        util[a] = v0
        node_v0 += strat[a] * v0

    # Regret + strategy-sum update for the acting player (value sign per player).
    reach_self = reach0 if player == 0 else reach1
    reach_opp = reach1 if player == 0 else reach0
    sign = 1.0 if player == 0 else -1.0
    node_util_self = sign * node_v0
    for a in actions:
        node.regret_sum[a] += reach_opp * (sign * util[a] - node_util_self)
        node.strategy_sum[a] += reach_self * strat[a]
    return node_v0


def solve(iterations: int = 20000, seed: int = 0) -> Dict[str, _Node]:
    """Run chance-sampled CFR and return the infoset -> node map."""
    rng = random.Random(seed)
    nodes: Dict[str, _Node] = {}
    for _ in range(iterations):
        deck = DECK[:]
        rng.shuffle(deck)
        c0, c1, board = deck[0], deck[1], deck[2]
        _cfr(nodes, (c0, c1, board), 0, "", [1, 1], 1.0, 1.0)
    return nodes


def _eval_v0(nodes, cards, round_idx, rhist, contrib) -> float:
    """Expected value to P0 of a node under the average strategy."""
    c0, c1, board = cards
    player = len(rhist) % 2
    if rhist.endswith("f"):
        folder = (len(rhist) - 1) % 2
        return contrib[1] if folder == 1 else -contrib[0]
    if _round_over(rhist):
        if round_idx == 0:
            return _eval_v0(nodes, cards, 1, "", contrib)
        win = _winner(c0, c1, board)
        return contrib[1] if win == 0 else (-contrib[0] if win == 1 else 0.0)
    to_call = max(contrib) - contrib[player]
    own_card = c0 if player == 0 else c1
    key = _infoset_key(own_card, board if round_idx == 1 else None, round_idx, rhist)
    node = nodes.get(key)
    actions = node.actions if node else _legal_actions(rhist, to_call)
    strat = node.average_strategy() if node else {a: 1.0 / len(actions) for a in actions}
    v0 = 0.0
    for a in actions:
        nc = list(contrib)
        if a == "c":
            nc[player] += to_call
        elif a == "r":
            nc[player] += to_call + BET_SIZE[round_idx]
        v0 += strat[a] * _eval_v0(nodes, cards, round_idx, rhist + a, nc)
    return v0


def state_value_table(nodes) -> Dict[Tuple[int, bool, int], float]:
    """Equilibrium value of holding a card at the start of a round, averaged over
    the opponent's card. Keyed by (own_rank, has_pair, round)."""
    raw: Dict[Tuple[int, bool, int], float] = {}
    # Round 0: no board yet; value is averaged over opponent card and board.
    for card in RANKS:
        vals = []
        for c1 in DECK_without(card):
            for board in DECK_without2(card, c1):
                vals.append(_eval_v0(nodes, (card, c1, board), 0, "", [1, 1]))
        raw[(card, False, 0)] = sum(vals) / len(vals)
    # Round 1: board known; split by whether we hold a pair.
    for card in RANKS:
        for board in RANKS:
            has_pair = card == board
            vals = []
            for c1 in DECK_without2(card, board):
                vals.append(_eval_v0(nodes, (card, c1, board), 1, "", [1, 1]))
            if vals:
                raw[(card, has_pair, 1)] = sum(vals) / len(vals)
    return raw


def DECK_without(card: int) -> List[int]:
    d = DECK[:]
    d.remove(card)
    return d


def DECK_without2(card: int, other: int) -> List[int]:
    d = DECK[:]
    d.remove(card)
    if other in d:
        d.remove(other)
    return d


@lru_cache(maxsize=1)
def potential_table(iterations: int = 20000) -> Dict[Tuple[int, bool, int], float]:
    """Normalised [0,1] CFR state-value potential, memoised per process."""
    nodes = solve(iterations)
    raw = state_value_table(nodes)
    lo, hi = min(raw.values()), max(raw.values())
    span = (hi - lo) or 1.0
    return {k: (v - lo) / span for k, v in raw.items()}


def _selftest() -> None:
    nodes = solve(iterations=int(os.environ.get("CFR_TEST_ITERS", "4000")), seed=1)

    # Qualitative Nash properties (robust even at low iteration counts):
    # 1) A pair of Kings on round 2, first to act, raises a lot.
    pair_kk = nodes.get(_infoset_key(2, 2, 1, ""))
    assert pair_kk is not None
    assert pair_kk.average_strategy()["r"] > 0.5, pair_kk.average_strategy()

    # 2) A Jack facing an opening raise in round 0 folds more than it raises.
    jack_vs_raise = nodes.get(_infoset_key(0, None, 0, "r"))
    assert jack_vs_raise is not None
    avs = jack_vs_raise.average_strategy()
    assert avs["f"] > avs.get("r", 0.0), avs

    # 3) State-value ordering: King-pair > bare King > bare Jack.
    tbl = state_value_table(nodes)
    assert tbl[(2, True, 1)] > tbl[(2, False, 1)] > tbl[(0, False, 1)], tbl
    print("cfr_leduc selftest OK")


if __name__ == "__main__":
    _selftest()
