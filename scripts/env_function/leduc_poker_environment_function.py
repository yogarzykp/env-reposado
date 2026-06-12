import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from threading import Lock, Semaphore

import requests
from trl.experimental.openenv import generate_rollout_completions

GAME_TO_TASK_ID_RANGE = {
    "goofspiel": (0, 99999999),
    "liars_dice": (100000000, 199999999),
    "leduc_poker": (200000000, 299999999),
    "gin_rummy": (300000000, 399999999),
    "othello": (400000000, 499999999),
    "backgammon": (500000000, 599999999),
    "hex": (600000000, 699999999),
    "clobber": (700000000, 799999999),
}

SELECTED_GAME = "leduc_poker"
REQUEST_TIMEOUT_SECONDS = 2400
INIT_TIMEOUT_SECONDS = 300
MAX_EPISODE_TOKENS = 8192
MAX_PROMPT_LEN = 8192 - 512

MCTS_CONFIG = {
    "opponent": "mcts",
    "mcts_max_simulations": 50,
    "mcts_num_rollouts": 1,
}

INVALID_ACTION_PENALTY = 0.10
NOOP_PENALTY = 0.03
TRUNCATION_PENALTY = 0.20
CONSECUTIVE_INVALID_ESCALATION = 0.05
SHAPING_REWARD_CLIP = 0.50
TERMINAL_REWARD_CLIP = 1.00

FOLD_WITH_PAIR_PENALTY = 0.08
CALL_WITH_NOTHING_PENALTY = 0.03
GOOD_RAISE_BONUS = 0.05
GOOD_FOLD_BONUS = 0.03
VALUE_BET_BONUS = 0.04
# BLUFF_SMALL_BONUS removed: temperature=0.0 at eval → bluff becomes deterministic
# → MCTS-50 exploits always-bluff-with-J pattern across all 2000 seeds.
# Source: PokerBench (AAAI 2025) — deterministic play cannot support mixed strategies.
RE_RAISE_WITH_PAIR_BONUS = 0.07   # Re-raise with pair: strongest value extraction play
CHECK_WITH_PAIR_PENALTY = 0.03
POT_ODDS_BONUS = 0.02

CARD_RANKS = {"J": 1, "Jack": 1, "Q": 2, "Queen": 2, "K": 3, "King": 3}

STRATEGY_TIPS = """
STRATEGY TIPS FOR LEDUC POKER:
- With a pair (your card matches the community card), you have the best possible hand — raise aggressively, and re-raise if opponent bets.
- With King high (no pair), you have a strong hand — consider raising or calling based on pot odds.
- With Jack high (no pair), you have the weakest hand — fold against aggression; calling is only correct if pot odds justify it.
- Play value-based poker: bet and raise when your hand is strong, fold or call conservatively when weak.
- In the first round (no community card yet), position matters — play tighter when acting first.
- Pot odds: if the pot is large relative to the call amount, calling with weaker hands becomes correct.
"""

REASONING_TAG_PAIRS = [
    ("think", "think"),
    ("thinking", "thinking"),
    ("reasoning", "reasoning"),
    ("thought", "thought"),
    ("reflection", "reflection"),
]

_ROLLOUT_STATE: dict = {}



class LeducBeliefState:

    FULL_DECK = ["J", "J", "Q", "Q", "K", "K"]
    BLUFF_PRIOR: float = 0.25

    def __init__(self) -> None:
        self._prob: dict[str, float] = {"J": 2 / 6, "Q": 2 / 6, "K": 2 / 6}
        self._raises_observed: int = 0
        self._known_cards: list[str] = []

    def set_known_cards(self, hole: str | None, board: str | None) -> None:
        self._known_cards = [c for c in [hole, board] if c in ("J", "Q", "K")]
        self._update_prior()

    def _update_prior(self) -> None:
        remaining = list(self.FULL_DECK)
        for c in self._known_cards:
            if c in remaining:
                remaining.remove(c)
        counts = {r: remaining.count(r) for r in ("J", "Q", "K")}
        total = sum(counts.values())
        self._prob = {r: (c / total if total > 0 else 1 / 3) for r, c in counts.items()}

    def _bayesian_update(self, likelihoods: dict[str, float]) -> None:
        unnorm = {r: self._prob.get(r, 0.0) * likelihoods.get(r, 0.5) for r in self._prob}
        total = sum(unnorm.values())
        if total > 0:
            self._prob = {r: v / total for r, v in unnorm.items()}

    def update_on_raise(self) -> None:
        self._raises_observed += 1
        self._bayesian_update({"J": self.BLUFF_PRIOR, "Q": 0.55, "K": 0.82})

    def update_on_check(self) -> None:
        self._bayesian_update({"J": 0.72, "Q": 0.52, "K": 0.30})

    def p_opp_has_pair(self, board: str | None) -> float:
        if board is None or board not in ("J", "Q", "K"):
            return 0.0
        return self._prob.get(board, 0.0)

    def p_opp_ahead(self, our_hole: str | None, board: str | None) -> float:
        if our_hole is None:
            return 0.5
        card_rank = {"J": 0, "Q": 1, "K": 2}
        we_have_pair = (board is not None and our_hole == board)
        if we_have_pair:
            return 0.0
        p_pair = self.p_opp_has_pair(board)
        our_rank = card_rank.get(our_hole, 0)
        p_higher = sum(
            p for r, p in self._prob.items()
            if card_rank.get(r, 0) > our_rank and r != board
        )
        return min(1.0, p_pair + p_higher * (1.0 - p_pair))

    def context_summary(self, our_hole: str | None, board: str | None) -> str:
        if self._raises_observed == 0:
            return ""
        p_ahead = self.p_opp_ahead(our_hole, board)
        p_pair = self.p_opp_has_pair(board)
        top = max(self._prob, key=self._prob.get)
        hint = (
            "→ Consider folding / checking (likely behind)"
            if p_ahead > 0.70 else
            "→ Consider raising (likely ahead)"
            if p_ahead < 0.30 else ""
        )
        return (
            f"[LeducBayes] Opp most likely hole: {top} ({self._prob[top]:.0%}) "
            f"| P(pair)={p_pair:.0%} P(ahead)={p_ahead:.0%}"
            + (f"  {hint}" if hint else "")
        )

    def reset(self, hole: str | None = None, board: str | None = None) -> None:
        self._prob = {"J": 2 / 6, "Q": 2 / 6, "K": 2 / 6}
        self._raises_observed = 0
        self._known_cards = []
        self.set_known_cards(hole, board)


def _is_truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))



def extract_and_format_observation(obs_text: str) -> str:
    return obs_text or ""


def _extract_legal_action_map(observation: str) -> dict[str, str]:
    if not observation:
        return {}
    match = re.search(
        r"Legal Actions:\s*\n(.*?)(?:\n\nYour choice|\nYour choice|\Z)",
        observation,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return {}

    block = match.group(1)
    mapping: dict[str, str] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "->" in line:
            left, right = line.split("->", 1)
            action_id = left.strip()
            label = right.strip()
        else:
            action_id = line.strip()
            label = action_id
        if re.fullmatch(r"-?\d+", action_id):
            mapping[action_id] = label
    return mapping


def _parse_card_rank(card_str: str) -> int | None:
    if not card_str:
        return None
    card_str = card_str.strip()
    for name, rank in CARD_RANKS.items():
        if name.lower() in card_str.lower():
            return rank
    return None


def _extract_state_features(observation: str) -> dict:
    features = {
        "private_card": None,
        "private_card_rank": None,
        "community_card": None,
        "community_card_rank": None,
        "has_pair": False,
        "round": 1,
        "pot_size": 0,
        "bet_to_call": 0,
        "num_raises_this_round": 0,
        "opponent_raised": False,
        "hand_strength": 0.0,
    }

    if not observation:
        return features

    private_match = re.search(
        r"(?:Your|Private|Hole)\s*(?:card|hand)\s*[:\s]+\s*(\w+)",
        observation, flags=re.IGNORECASE
    )
    if private_match:
        features["private_card"] = private_match.group(1)
        features["private_card_rank"] = _parse_card_rank(private_match.group(1))

    community_match = re.search(
        r"(?:Community|Public|Board|Flop)\s*(?:card)?\s*[:\s]+\s*(\w+)",
        observation, flags=re.IGNORECASE
    )
    if community_match:
        card_text = community_match.group(1).strip()
        if card_text.lower() not in ("none", "n/a", "not", "no"):
            features["community_card"] = card_text
            features["community_card_rank"] = _parse_card_rank(card_text)
            features["round"] = 2

    if (features["private_card_rank"] is not None and
            features["community_card_rank"] is not None and
            features["private_card_rank"] == features["community_card_rank"]):
        features["has_pair"] = True

    pot_match = re.search(r"(?:Pot|Total pot)\s*[:\s]+\s*(\d+)", observation, flags=re.IGNORECASE)
    if pot_match:
        features["pot_size"] = int(pot_match.group(1))

    call_match = re.search(
        r"(?:to call|call amount|bet)\s*[:\s]+\s*(\d+)",
        observation, flags=re.IGNORECASE
    )
    if call_match:
        features["bet_to_call"] = int(call_match.group(1))

    raises = len(re.findall(r"(?:raise|Raise|RAISE)", observation))
    features["num_raises_this_round"] = raises

    if re.search(r"(?:opponent|player\s*\d)\s+(?:raised|raises|bet|bets)", observation, flags=re.IGNORECASE):
        features["opponent_raised"] = True

    features["hand_strength"] = _calculate_hand_strength(features)

    return features


def _calculate_hand_strength(features: dict) -> float:
    private_rank = features.get("private_card_rank")
    if private_rank is None:
        return 0.5

    if features.get("has_pair"):
        return 0.8 + (private_rank - 1) * 0.1

    if features.get("round") == 2:
        return private_rank * 0.17
    else:
        return 0.1 + private_rank * 0.2



def _classify_action(label: str) -> str:
    low = (label or "").strip().lower()
    if "fold" in low:
        return "fold"
    if "check" in low:
        return "check"
    if "call" in low:
        return "call"
    if "raise" in low:
        return "raise"
    if "bet" in low:
        return "bet"
    return "unknown"


def _parse_action_id(
    completion_text: str,
    legal_action_map: dict[str, str],
) -> str:
    if not legal_action_map:
        return ""

    cleaned = remove_reasoning_tags(completion_text or "")
    if cleaned.endswith("</s>"):
        cleaned = cleaned[:-5]
    if "Action:" in cleaned:
        cleaned = cleaned.split("Action:")[-1].strip()

    for num in re.findall(r"-?\d+", cleaned):
        if num in legal_action_map:
            return num

    normalized = cleaned.strip().lower()
    for action_id, label in legal_action_map.items():
        if normalized == label.strip().lower():
            return action_id

    for keyword in ("fold", "check", "call", "raise", "bet"):
        if keyword in normalized:
            for action_id, label in legal_action_map.items():
                if keyword in label.strip().lower():
                    return action_id

    return ""



def _compute_shaping_reward(
    state_features: dict,
    action_label: str,
) -> tuple[float, dict]:

    action_type = _classify_action(action_label)
    hand_strength = state_features.get("hand_strength", 0.5)
    has_pair = state_features.get("has_pair", False)
    opponent_raised = state_features.get("opponent_raised", False)
    pot_size = state_features.get("pot_size", 0)
    bet_to_call = state_features.get("bet_to_call", 0)
    round_num = state_features.get("round", 1)

    # Round-specific reward weight (AT-GRPO insight):
    # Round 2 decisions are more informed (board card revealed) → amplify positive signal.
    # Round 1 decisions have higher uncertainty → baseline weight.
    # Applied ONLY to positive shaping to avoid disproportionate penalty amplification.
    ROUND_WEIGHT = 1.3 if round_num == 2 else 1.0

    shaping = 0.0
    meta = {
        "action_type": action_type,
        "hand_strength": hand_strength,
        "has_pair": has_pair,
        "round": round_num,
        "round_weight": ROUND_WEIGHT,
        "shaping_components": [],
    }

    if action_type == "fold":
        if has_pair:
            shaping -= FOLD_WITH_PAIR_PENALTY
            meta["shaping_components"].append("fold_with_pair_penalty")
        elif hand_strength >= 0.5 and not opponent_raised:
            shaping -= FOLD_WITH_PAIR_PENALTY * 0.3
            meta["shaping_components"].append("fold_decent_hand")
        elif hand_strength < 0.3 and opponent_raised:
            shaping += GOOD_FOLD_BONUS
            meta["shaping_components"].append("good_fold_weak_hand")

    elif action_type in ("raise", "bet"):
        if has_pair and opponent_raised:
            # Re-raise with pair: strongest value extraction decision in Leduc.
            # Opponent already raised → our pair is near-certain winner → must maximize pot.
            # Source: DeepStack hand strength valuation + GiGPO step-level credit.
            shaping += RE_RAISE_WITH_PAIR_BONUS
            meta["shaping_components"].append("reraise_with_pair")
        elif has_pair:
            # Standard value bet with pair (no prior raise from opponent)
            shaping += VALUE_BET_BONUS
            meta["shaping_components"].append("value_bet_pair")
        elif hand_strength >= 0.5:
            shaping += GOOD_RAISE_BONUS * 0.5
            meta["shaping_components"].append("raise_strong_hand")
        # NOTE: bluff_bonus removed — temperature=0.0 at eval makes bluffing
        # deterministically exploitable by MCTS-50. No bonus for raise with weak hand.

    elif action_type == "call":
        if has_pair:
            shaping += 0.01
            meta["shaping_components"].append("call_with_pair")
        elif opponent_raised and hand_strength < 0.25:
            shaping -= CALL_WITH_NOTHING_PENALTY
            meta["shaping_components"].append("bad_call_weak")
        elif pot_size > 0 and bet_to_call > 0:
            pot_odds = bet_to_call / (pot_size + bet_to_call) if (pot_size + bet_to_call) > 0 else 1.0
            if hand_strength >= pot_odds:
                shaping += POT_ODDS_BONUS
                meta["shaping_components"].append("pot_odds_correct")

    elif action_type == "check":
        if has_pair and round_num == 2:
            shaping -= CHECK_WITH_PAIR_PENALTY
            meta["shaping_components"].append("check_with_pair_miss_value")

    # Apply round weight: amplify POSITIVE signal for R2 (informed decisions),
    # keep penalties at face value across both rounds.
    if shaping > 0:
        shaping *= ROUND_WEIGHT
        meta["shaping_components"].append(f"round_weight_x{ROUND_WEIGHT}")

    return shaping, meta




def _select_fallback_action(
    legal_action_map: dict[str, str],
    state_features: dict,
) -> str:
    has_pair = state_features.get("has_pair", False)
    hand_strength = state_features.get("hand_strength", 0.5)

    for action_id, label in legal_action_map.items():
        if "check" in label.lower():
            return action_id

    if has_pair or hand_strength >= 0.5:
        for action_id, label in legal_action_map.items():
            if "call" in label.lower():
                return action_id

    for action_id, label in legal_action_map.items():
        if "fold" in label.lower():
            return action_id

    return sorted(legal_action_map.keys(), key=lambda x: int(x))[0]



def _extract_terminal_reward(step_block: dict, observation_text: str) -> float:
    info = step_block.get("info", {}) if isinstance(step_block, dict) else {}

    cumulative_reward = info.get("cumulative_reward")
    if isinstance(cumulative_reward, (int, float)):
        return _clamp(float(cumulative_reward), -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)

    your_return_match = re.search(r"Your Return:\s*([+-]?\d+(?:\.\d+)?)", observation_text or "")
    if your_return_match:
        return _clamp(float(your_return_match.group(1)), -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)

    normalized_match = re.search(r"Normalized Score:\s*([+-]?\d+(?:\.\d+)?)", observation_text or "")
    result_match = re.search(r"Result:\s*(WIN|LOSS|DRAW)", observation_text or "", flags=re.IGNORECASE)
    if normalized_match:
        normalized_value = float(normalized_match.group(1))
        if result_match:
            result = result_match.group(1).upper()
            if result == "LOSS":
                normalized_value = -abs(normalized_value) if normalized_value != 0 else -1.0
            elif result == "WIN":
                normalized_value = abs(normalized_value) if normalized_value != 0 else 1.0
            else:
                normalized_value = 0.0
        return _clamp(normalized_value, -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)

    step_reward = _safe_float(step_block.get("reward", 0.0), default=0.0)
    return _clamp(step_reward, -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)



class EpisodeTraceLogger:

    def __init__(self, trace_dir: str, rank: int):
        self.trace_dir = trace_dir
        self.rank = rank
        self._lock = Lock()
        self.log_path = os.path.join(self.trace_dir, f"leduc_poker_episode_traces_rank{rank}.jsonl")
        self.max_text_chars = int(os.environ.get("EPISODE_TRACE_MAX_TEXT_CHARS", "4000"))
        self.sample_rate = float(os.environ.get("EPISODE_TRACE_SAMPLE_RATE", "1.0"))

        os.makedirs(self.trace_dir, exist_ok=True)
        print(f"[EPISODE_TRACE] Writing traces to {self.log_path}")

    def should_log(self) -> bool:
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        return random.random() <= self.sample_rate

    def clip_text(self, text: str) -> str:
        if not text:
            return ""
        if len(text) <= self.max_text_chars:
            return text
        return text[: self.max_text_chars] + f"... [truncated {len(text) - self.max_text_chars} chars]"

    def log_episode(self, payload: dict) -> None:
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True) + "\n")


class CurriculumScheduler:

    def __init__(
        self,
        initial_max_turn: int = 2,
        final_max_turn: int = 10,
        rollouts_per_stage: int = 1280,
        initial_hint_prob: float = 0.0,
        final_hint_prob: float = 0.0,
        warmup_rollouts: int = 128,
    ):
        self.initial_max_turn = initial_max_turn
        self.final_max_turn = final_max_turn
        self.rollouts_per_stage = rollouts_per_stage
        self.initial_hint_prob = initial_hint_prob
        self.final_hint_prob = final_hint_prob
        self.warmup_rollouts = warmup_rollouts
        self.total_rollouts = 0

    def get_max_turn(self) -> int:
        if self.total_rollouts < self.warmup_rollouts:
            return self.initial_max_turn
        adjusted_rollouts = self.total_rollouts - self.warmup_rollouts
        stage = adjusted_rollouts // self.rollouts_per_stage
        return min(self.initial_max_turn + stage, self.final_max_turn)

    def get_hint_prob(self) -> float:
        if self.total_rollouts < self.warmup_rollouts:
            return self.initial_hint_prob
        total_stages = max(self.final_max_turn - self.initial_max_turn, 1)
        total_decay_rollouts = total_stages * self.rollouts_per_stage
        adjusted_rollouts = self.total_rollouts - self.warmup_rollouts
        progress = min(adjusted_rollouts / total_decay_rollouts, 1.0)
        current_prob = self.initial_hint_prob - progress * (self.initial_hint_prob - self.final_hint_prob)
        return max(current_prob, self.final_hint_prob)

    def step(self, num_rollouts: int = 1) -> None:
        self.total_rollouts += num_rollouts



def remove_reasoning_tags(text: str) -> str:
    cleaned = text
    for tag_name, close_name in REASONING_TAG_PAIRS:
        cleaned = re.sub(
            rf"<{tag_name}>.*?</{close_name}>",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        close_tag = f"</{close_name}>"
        if close_tag in cleaned:
            cleaned = cleaned.split(close_tag)[-1]
        open_match = re.search(rf"<{tag_name}>", cleaned, flags=re.IGNORECASE)
        if open_match:
            cleaned = cleaned[: open_match.start()]
    cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)
    return cleaned.strip()



def _get_system_prompt(use_hints: bool) -> str:
    system_prompt = """You are playing leduc_poker.

# Game Rules
LEDUC POKER RULES:

Setup: A simplified poker game using a deck of 6 cards: 2 Jacks, 2 Queens, 2 Kings.
Each player is dealt one private hole card. Both players ante 1 chip to start.

Round 1 (Pre-flop):
- Players bet based only on their private card.
- Actions: Check, Bet/Raise, Call, Fold.
- Fixed bet size: 2 chips. Maximum 2 raises per round.

Round 2 (Post-flop):
- A single community card is revealed face-up.
- Another betting round occurs with larger bets: 4 chips. Maximum 2 raises per round.

Showdown:
- If your private card matches the community card rank, you have a PAIR (strongest hand).
- If neither player has a pair, the higher-ranked card wins (King > Queen > Jack).

# Output Format
You must respond with ONLY the action ID (a single number).
Do NOT include descriptions or explanations.
Examples:
- For action "0 -> Fold": respond "0"
- For action "1 -> Call": respond "1"
- For action "2 -> Raise": respond "2"
"""
    if use_hints:
        system_prompt += "\n" + STRATEGY_TIPS
    return system_prompt



def _build_env_pool(server_urls: list[str]) -> list[dict[str, str]]:
    env_pool = []
    init_task_id = GAME_TO_TASK_ID_RANGE[SELECTED_GAME][0]

    for idx, base_url in enumerate(server_urls):
        try:
            print(f"[INIT] Initializing env on server {idx}: {base_url}")
            payload = {"task_id": init_task_id, "seed": 42, **MCTS_CONFIG}
            res = requests.post(f"{base_url}/reset", json=payload, timeout=INIT_TIMEOUT_SECONDS)
            res.raise_for_status()
            env_pool.append({"base_url": base_url})
            print(f"[INIT] Server {idx} ready")
        except Exception as e:
            raise RuntimeError(f"Failed to init server {base_url}: {e}") from e

    return env_pool


def _initialize_rollout_state(trainer) -> None:
    if _ROLLOUT_STATE.get("initialized", False):
        return

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
    server_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]
    if not server_urls:
        raise RuntimeError("ENVIRONMENT_SERVER_URLS is empty")

    env_pool = _build_env_pool(server_urls)
    rollout_per_stage = int(getattr(trainer.args, "rollouts_per_stage", 1280))
    initial_max_turn = int(getattr(trainer.args, "initial_max_turn", 2))
    final_max_turn = int(os.environ.get("LEDUC_POKER_FINAL_MAX_TURN", "10"))
    initial_hint_prob = float(os.environ.get("LEDUC_POKER_INITIAL_HINT_PROB", "0.0"))
    final_hint_prob = float(os.environ.get("LEDUC_POKER_FINAL_HINT_PROB", "0.0"))

    # Read warmup from grpo_env_config (via trainer.args), matching GR/LD pattern
    rollout_warmup_rollouts = (
        trainer.args.rollout_warmup_rollouts
        if getattr(trainer.args, "rollout_warmup_rollouts", None) is not None
        else rollout_per_stage  # fallback: 1 full stage
    )

    _ROLLOUT_STATE["rank"] = rank
    _ROLLOUT_STATE["env_pool"] = env_pool
    _ROLLOUT_STATE["num_servers"] = len(env_pool)
    _ROLLOUT_STATE["thread_pool"] = ThreadPoolExecutor(max_workers=len(env_pool))
    _ROLLOUT_STATE["generation_semaphore"] = Semaphore(1)
    _ROLLOUT_STATE["curriculum"] = CurriculumScheduler(
        initial_max_turn=initial_max_turn,
        final_max_turn=final_max_turn,
        rollouts_per_stage=rollout_per_stage,
        initial_hint_prob=initial_hint_prob,
        final_hint_prob=final_hint_prob,
        warmup_rollouts=rollout_warmup_rollouts,
    )
    _ROLLOUT_STATE["initialized"] = True

    if rank == 0:
        print(
            f"[LEDUC_POKER] Initialized: max_turn={initial_max_turn}→{final_max_turn}, "
            f"rollouts_per_stage={rollout_per_stage}, "
            f"warmup_rollouts={rollout_warmup_rollouts}, "
            f"hint_prob={initial_hint_prob}→{final_hint_prob}, "
            f"MCTS={MCTS_CONFIG['mcts_max_simulations']}/{MCTS_CONFIG['mcts_num_rollouts']}"
        )

    trace_enabled = _is_truthy_env(os.environ.get("EPISODE_TRACE_ENABLED", "1"))
    trace_dir = os.environ.get("EPISODE_TRACE_DIR", "").strip()
    _ROLLOUT_STATE["trace_logger"] = None
    if trace_enabled and trace_dir:
        try:
            _ROLLOUT_STATE["trace_logger"] = EpisodeTraceLogger(trace_dir=trace_dir, rank=rank)
        except Exception as e:
            print(f"[EPISODE_TRACE] Failed to initialize logger: {e}")
    elif rank == 0:
        print("[EPISODE_TRACE] Disabled (set EPISODE_TRACE_ENABLED=1 and EPISODE_TRACE_DIR)")


def _reset_environment(env_endpoint: str, game_id: int, timeout: int) -> tuple[str, str]:
    payload = {"task_id": game_id, "seed": random.randint(0, 2**31 - 1), **MCTS_CONFIG}
    reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=timeout)
    reset_res.raise_for_status()
    reset_data = reset_res.json()
    result_block = reset_data["result"]
    episode_id = result_block.get("episode_id", "")
    raw_observation = result_block.get("observation", "")
    return episode_id, extract_and_format_observation(raw_observation)


def _step_environment(
    env_endpoint: str,
    episode_id: str,
    action_to_send: str,
    timeout: int,
) -> tuple[str, float, bool, dict]:
    step_payload = {"action": action_to_send, "episode_id": episode_id}
    step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=timeout)
    step_res.raise_for_status()
    step_data = step_res.json()
    step_block = step_data["result"]
    raw_observation = step_block.get("observation", "")
    formatted_observation = extract_and_format_observation(raw_observation)
    step_reward = _safe_float(step_block.get("reward", 0.0), default=0.0)
    done = bool(step_block.get("done", False))
    return formatted_observation, step_reward, done, step_block



def _last_prompt_fallback_result() -> dict:
    return {
        "prompt_ids": [1],
        "completion_ids": [1],
        "logprobs": [1.0],
        "reward": 0.0,
        "final_score": 0.0,
    }


def _full_prompt_fallback_result() -> dict:
    return {
        "prompt_ids": [1],
        "completion_ids": [1],
        "action_mask": [0],
        "logprobs": [1.0],
        "reward": 0.0,
        "final_score": 0.0,
    }


def _execute_parallel_rollouts(prompts, executor, run_single_prompt, fallback_builder):
    results = [None] * len(prompts)
    futures = [executor.submit(run_single_prompt, i, p) for i, p in enumerate(prompts)]

    for future in as_completed(futures):
        idx, res = future.result()
        results[idx] = res if res is not None else fallback_builder()

    return [r for r in results if r is not None]


def _log_batch_statistics(list_results: list[dict]) -> None:
    finished = sum(1 for r in list_results if r["final_score"] != 0)
    avg_return = sum(r["reward"] for r in list_results) / len(list_results) if list_results else 0.0
    wins = sum(1 for r in list_results if r["final_score"] > 0)
    print(f"[BATCH] Finished: {finished}/{len(list_results)}, Wins: {wins}, AvgReturn: {avg_return:.3f}")



def _rollout_parallelized_curriculum(
    prompts: list[str],
    trainer,
    include_action_mask: bool,
) -> dict[str, list]:
    _initialize_rollout_state(trainer)

    rank = _ROLLOUT_STATE["rank"]
    env_pool = _ROLLOUT_STATE["env_pool"]
    num_servers = _ROLLOUT_STATE["num_servers"]
    curriculum: CurriculumScheduler = _ROLLOUT_STATE["curriculum"]
    trace_logger = _ROLLOUT_STATE["trace_logger"]

    tokenizer = trainer.processing_class
    timeout = REQUEST_TIMEOUT_SECONDS
    current_max_turn = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()
    print(
        f"[CURRICULUM] Rollout {curriculum.total_rollouts}: "
        f"max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}"
    )

    def run_single_prompt(index: int, prompt: str):
        game_id = int(prompt)
        server_idx = (index + rank) % num_servers
        server = env_pool[server_idx]
        env_endpoint = server["base_url"]

        invalid_count = 0
        consecutive_invalids = 0
        noop_count = 0
        consecutive_noops = 0
        done = False
        final_reward = 0.0
        turn_number = 0
        accumulated_shaping_reward = 0.0
        step_records = []
        termination_reason = "unknown"
        last_step_block: dict = {}

        belief = LeducBeliefState()

        if include_action_mask:
            episode_prompt_ids: list[int] = []
            episode_completion_ids: list[int] = []
            episode_logprobs: list[float] = []
            episode_action_mask: list[int] = []
            prev_full_ids: list[int] | None = None
        else:
            prompt_ids_last: list[int] = []
            completion_ids_last: list[int] = []
            logprobs_last: list[float] = []

        try:
            episode_id, formatted_observation = _reset_environment(
                env_endpoint=env_endpoint,
                game_id=game_id,
                timeout=timeout,
            )
        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            if trace_logger and trace_logger.should_log():
                trace_logger.log_episode(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "game_id": game_id,
                        "status": "reset_failed",
                        "error": str(e),
                    }
                )
            return index, None

        use_hints = random.random() < current_hint_prob
        init_state = _extract_state_features(formatted_observation)
        belief.set_known_cards(
            init_state.get("private_card"),
            init_state.get("community_card"),
        )
        messages = [
            {"role": "system", "content": _get_system_prompt(use_hints=use_hints)},
            {"role": "user", "content": formatted_observation},
        ]

        while not done and turn_number < current_max_turn:
            observation_before_action = formatted_observation
            legal_action_map = _extract_legal_action_map(observation_before_action)
            state_features = _extract_state_features(observation_before_action)

            if not legal_action_map:
                accumulated_shaping_reward -= INVALID_ACTION_PENALTY
                termination_reason = "no_legal_actions"
                break

            with _ROLLOUT_STATE["generation_semaphore"]:
                rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]

            prompt_ids = rollout_outputs.get("prompt_ids", [])
            completion_ids = rollout_outputs.get("completion_ids", [])
            logprobs = rollout_outputs.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            if include_action_mask:
                if len(prompt_ids) > MAX_PROMPT_LEN:
                    print(
                        f"Warning: Prompt exceeded {MAX_PROMPT_LEN} tokens ({len(prompt_ids)}) at turn {turn_number}"
                    )
                    termination_reason = "max_prompt_len_exceeded"
                    break

                if turn_number == 0:
                    episode_prompt_ids = prompt_ids
                    prev_full_ids = prompt_ids.copy()
                else:
                    if prev_full_ids is None:
                        prev_full_ids = prompt_ids.copy()
                    elif prompt_ids[: len(prev_full_ids)] != prev_full_ids:
                        prev_full_ids = prompt_ids.copy()
                    else:
                        delta_prompt_ids = prompt_ids[len(prev_full_ids) :]
                        if delta_prompt_ids:
                            episode_completion_ids.extend(delta_prompt_ids)
                            episode_logprobs.extend([0.0] * len(delta_prompt_ids))
                            episode_action_mask.extend([0] * len(delta_prompt_ids))
                        prev_full_ids = prompt_ids.copy()

                if completion_ids:
                    episode_completion_ids.extend(completion_ids)
                    episode_logprobs.extend(logprobs)
                    episode_action_mask.extend([1] * len(completion_ids))
                    if prev_full_ids is not None:
                        prev_full_ids = prev_full_ids + completion_ids
            else:
                prompt_ids_last = prompt_ids
                completion_ids_last = completion_ids
                logprobs_last = logprobs

            messages.append({"role": "assistant", "content": completion_text})

            action_to_send = _parse_action_id(completion_text, legal_action_map)
            parse_failed = not action_to_send
            if parse_failed or action_to_send not in legal_action_map:
                invalid_count += 1
                consecutive_invalids += 1
                penalty = INVALID_ACTION_PENALTY + CONSECUTIVE_INVALID_ESCALATION * (consecutive_invalids - 1)
                accumulated_shaping_reward -= penalty
                action_to_send = _select_fallback_action(legal_action_map, state_features)
            else:
                consecutive_invalids = 0

            action_label = legal_action_map.get(action_to_send, "")
            action_type = _classify_action(action_label)

            if state_features.get("opponent_raised"):
                belief.update_on_raise()
            else:
                belief.update_on_check()
            belief.set_known_cards(
                state_features.get("private_card"),
                state_features.get("community_card"),
            )

            shaping_reward, shaping_meta = _compute_shaping_reward(state_features, action_label)
            accumulated_shaping_reward += shaping_reward

            try:
                formatted_observation, step_reward, done, last_step_block = _step_environment(
                    env_endpoint=env_endpoint,
                    episode_id=episode_id,
                    action_to_send=action_to_send,
                    timeout=timeout,
                )
            except Exception as e:
                print(f"Step failed: {e}")
                formatted_observation = ""
                step_reward = -0.01
                done = False
                invalid_count += 1
                accumulated_shaping_reward -= INVALID_ACTION_PENALTY
                last_step_block = {"reward": step_reward, "done": False}

            observation_lower = formatted_observation.lower()
            invalid_or_noop = (
                "invalid" in observation_lower
                or "nothing happens" in observation_lower
                or "nothing happened" in observation_lower
                or action_to_send not in legal_action_map
            )
            if invalid_or_noop:
                invalid_count += 1
                accumulated_shaping_reward -= INVALID_ACTION_PENALTY

            if formatted_observation == observation_before_action:
                noop_count += 1
                consecutive_noops += 1
                accumulated_shaping_reward -= NOOP_PENALTY
                if consecutive_noops >= 3:
                    accumulated_shaping_reward -= NOOP_PENALTY * (consecutive_noops - 2)
            else:
                consecutive_noops = 0

            if done:
                final_reward = _extract_terminal_reward(last_step_block, formatted_observation)
                termination_reason = "done"
            else:
                next_state = _extract_state_features(formatted_observation)
                bayes_ctx = belief.context_summary(
                    next_state.get("private_card"),
                    next_state.get("community_card"),
                )
                obs_augmented = (
                    formatted_observation + "\n\n" + bayes_ctx
                    if bayes_ctx else formatted_observation
                )
                messages.append({"role": "user", "content": obs_augmented})

            step_records.append(
                {
                    "turn": turn_number,
                    "assistant_text": trace_logger.clip_text(completion_text) if trace_logger else completion_text,
                    "parsed_action": action_to_send,
                    "action_label": action_label,
                    "action_type": shaping_meta.get("action_type", "unknown"),
                    "hand_strength": shaping_meta.get("hand_strength", 0.0),
                    "has_pair": shaping_meta.get("has_pair", False),
                    "shaping_reward": float(shaping_reward),
                    "shaping_components": shaping_meta.get("shaping_components", []),
                    "observation_before_action": (
                        trace_logger.clip_text(observation_before_action)
                        if trace_logger
                        else observation_before_action
                    ),
                    "observation_after_action": (
                        trace_logger.clip_text(formatted_observation) if trace_logger else formatted_observation
                    ),
                    "step_reward": float(step_reward),
                    "done": bool(done),
                    "invalid_or_noop": invalid_or_noop,
                    "parse_failed": bool(parse_failed),
                }
            )

            turn_number += 1

        if not done:
            if termination_reason == "unknown":
                termination_reason = "max_turn_reached"
            if current_max_turn < curriculum.final_max_turn:
                final_reward = 0.0
            else:
                final_reward = -TRUNCATION_PENALTY
            accumulated_shaping_reward -= TRUNCATION_PENALTY

        clipped_shaping = _clamp(accumulated_shaping_reward, -SHAPING_REWARD_CLIP, SHAPING_REWARD_CLIP)
        train_reward = _clamp(final_reward + clipped_shaping, -1.0, 1.0)

        print(
            f"[ID:{game_id} Done:{int(done)} T:{turn_number:2d} "
            f"Env:{final_reward:+.3f} Shape:{accumulated_shaping_reward:+.3f} "
            f"ClipShape:{clipped_shaping:+.3f} Inv:{invalid_count}"
        )

        if trace_logger and trace_logger.should_log():
            trace_logger.log_episode(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "game_id": game_id,
                    "episode_id": episode_id,
                    "environment": "leduc_poker",
                    "status": "completed" if done else "truncated",
                    "termination_reason": termination_reason,
                    "turns": turn_number,
                    "final_reward": float(final_reward),
                    "raw_shaping_reward": float(accumulated_shaping_reward),
                    "clipped_shaping_reward": float(clipped_shaping),
                    "train_reward": float(train_reward),
                    "invalid_count": invalid_count,
                    "steps": step_records,
                }
            )

        if include_action_mask:
            if len(episode_completion_ids) > MAX_EPISODE_TOKENS:
                episode_completion_ids = episode_completion_ids[:MAX_EPISODE_TOKENS]
                episode_logprobs = episode_logprobs[:MAX_EPISODE_TOKENS]
                episode_action_mask = episode_action_mask[:MAX_EPISODE_TOKENS]

            return index, {
                "prompt_ids": episode_prompt_ids,
                "completion_ids": episode_completion_ids,
                "action_mask": episode_action_mask,
                "logprobs": episode_logprobs,
                "reward": train_reward,
                "final_score": final_reward,
            }

        return index, {
            "prompt_ids": prompt_ids_last,
            "completion_ids": completion_ids_last,
            "logprobs": logprobs_last,
            "reward": train_reward,
            "final_score": final_reward,
        }

    executor = _ROLLOUT_STATE["thread_pool"]
    fallback_builder = _full_prompt_fallback_result if include_action_mask else _last_prompt_fallback_result
    list_results = _execute_parallel_rollouts(
        prompts=prompts,
        executor=executor,
        run_single_prompt=run_single_prompt,
        fallback_builder=fallback_builder,
    )

    curriculum.step(len(prompts))
    _log_batch_statistics(list_results)

    if include_action_mask:
        return {
            "prompt_ids": [r["prompt_ids"] for r in list_results],
            "completion_ids": [r["completion_ids"] for r in list_results],
            "action_mask": [r["action_mask"] for r in list_results],
            "logprobs": [r["logprobs"] for r in list_results],
            "env_rewards": [r["reward"] for r in list_results],
        }

    return {
        "prompt_ids": [r["prompt_ids"] for r in list_results],
        "completion_ids": [r["completion_ids"] for r in list_results],
        "logprobs": [r["logprobs"] for r in list_results],
        "env_rewards": [r["reward"] for r in list_results],
    }



def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    del max_turns
    return _rollout_parallelized_curriculum(prompts=prompts, trainer=trainer, include_action_mask=False)


def rollout_full_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    del max_turns
    return _rollout_parallelized_curriculum(prompts=prompts, trainer=trainer, include_action_mask=True)


def rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)
