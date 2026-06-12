"""Game registry: binds each tournament game to its prompt, feature parser and
potential function, so the ReST pipeline (collector / filter / cot / trainer) is
game-agnostic and a new game is a plug-in entry rather than a code change.

Output contract is the same Thought/Action format across all games (the
validator's reasoning-capable parser). The per-game differences are the rules
text, the state features used for shaping, and the potential Phi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import shaping
from selfplay.rollout_collector import LEDUC_SYSTEM_PROMPT

_THOUGHT_ACTION_FOOTER = (
    "\nOn each turn you receive the state and a list of legal actions written as "
    "`<id> -> <name>`. Respond in exactly this format and nothing else:\n"
    "Thought:\n<one or two sentences of reasoning>\n\nAction:\n<the integer id of one legal action>\n"
)

GOOFSPIEL_SYSTEM_PROMPT = (
    "You are playing Goofspiel. Each player holds bid cards 1..N; a shuffled prize "
    "deck of 1..N is revealed one card at a time. Both players secretly play a bid "
    "card; the higher bid wins the revealed prize's points (a tie discards it). The "
    "player with the most points at the end wins. Spend high cards on high prizes "
    "and low cards on low prizes." + _THOUGHT_ACTION_FOOTER
)

LIARS_DICE_SYSTEM_PROMPT = (
    "You are playing Liar's Dice. Each player has hidden dice; players take turns "
    "raising a bid of the form (quantity, face) over all dice on the table, or "
    "calling the previous bidder a liar. If a challenged bid is met or exceeded the "
    "bidder wins, otherwise the challenger wins. Sixes are wild. Bid when your own "
    "dice make the claim plausible; challenge when the standing bid is unlikely."
    + _THOUGHT_ACTION_FOOTER
)

GIN_RUMMY_SYSTEM_PROMPT = (
    "You are playing Gin Rummy. Form melds (runs of 3+ in a suit, or sets of 3-4 of "
    "a rank) to minimise your deadwood (the value of unmatched cards). Draw the "
    "upcard only when it improves a meld; otherwise draw from the stock. Knock as "
    "soon as your deadwood is low enough, and knock early when ahead."
    + _THOUGHT_ACTION_FOOTER
)


@dataclass
class GameSpec:
    name: str
    task_id_range: Tuple[int, int]
    system_prompt: str
    features_fn: Callable[[str], Dict[str, object]]
    potential_fn: Callable[[Dict[str, object]], float]

    @property
    def game_id(self) -> int:
        return self.task_id_range[0]


GAME_SPECS: Dict[str, GameSpec] = {
    "goofspiel": GameSpec(
        "goofspiel", (0, 99999999), GOOFSPIEL_SYSTEM_PROMPT,
        shaping.goofspiel_features, shaping.potential_goofspiel),
    "liars_dice": GameSpec(
        "liars_dice", (100000000, 199999999), LIARS_DICE_SYSTEM_PROMPT,
        shaping.liars_dice_features, shaping.potential_liars_dice),
    "leduc_poker": GameSpec(
        "leduc_poker", (200000000, 299999999), LEDUC_SYSTEM_PROMPT,
        shaping.leduc_features, shaping.potential_leduc),
    "gin_rummy": GameSpec(
        "gin_rummy", (300000000, 399999999), GIN_RUMMY_SYSTEM_PROMPT,
        shaping.gin_rummy_features, shaping.potential_gin_rummy),
}


def get_spec(name: str) -> GameSpec:
    if name not in GAME_SPECS:
        raise KeyError(f"unknown game '{name}'; known: {list(GAME_SPECS)}")
    return GAME_SPECS[name]


def get_spec_by_task_id(task_id: int) -> GameSpec:
    for spec in GAME_SPECS.values():
        lo, hi = spec.task_id_range
        if lo <= task_id <= hi:
            return spec
    raise ValueError(f"task_id {task_id} matches no game range")


def _selftest() -> None:
    assert get_spec("leduc_poker").game_id == 200000000
    assert get_spec_by_task_id(150000000).name == "liars_dice"
    assert get_spec_by_task_id(350000000).name == "gin_rummy"
    for spec in GAME_SPECS.values():
        assert "Thought:" in spec.system_prompt and "Action:" in spec.system_prompt
        assert 0.0 <= spec.potential_fn(spec.features_fn("")) <= 1.0  # empty -> 0.5
    print("game_spec selftest OK")


if __name__ == "__main__":
    _selftest()
