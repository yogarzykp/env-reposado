import json
import math
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from threading import Lock, Semaphore

import requests
from scipy.stats import binom as _binom
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

SELECTED_GAME = "liars_dice"
REQUEST_TIMEOUT_SECONDS = 2400
INIT_TIMEOUT_SECONDS = 300
MAX_EPISODE_TOKENS = 16384
MAX_PROMPT_LEN = 16384 - 512

MCTS_CONFIG = {
    "opponent": "mcts",
    "mcts_max_simulations": 225,  # updated to match tournament eval (April 2026)
    "mcts_num_rollouts": 1,
}

# Curriculum MCTS settings (ramp from easy to overshoot tournament level)
INITIAL_MCTS_SIMS = 50    # start at previous training level
FINAL_MCTS_SIMS = 300     # overshoot tournament eval (225) by ~33%
MCTS_WARMUP_STEPS = 20    # optimizer steps to ramp from initial to final

# Reward settings
INVALID_ACTION_PENALTY = 0.10
NOOP_PENALTY = 0.03
TRUNCATION_PENALTY = 0.20
CALL_QUALITY_BONUS = 0.08
CALL_QUALITY_PENALTY = 0.06
BID_PLAUSIBILITY_BONUS = 0.04
BID_PLAUSIBILITY_PENALTY = 0.04
SHAPING_REWARD_CLIP = 0.50
TERMINAL_REWARD_CLIP = 1.00

# Self-play RL bid quality constants (from WiLDCARD / Stanford CFR literature)
# Optimal strategy: small incremental bids are harder to challenge and more strategically sound
BID_MARGINAL_BONUS = 0.02      # bonus for minimal increment (+1 qty or +1 face) — incremental = optimal
BID_AGGRESSIVE_PENALTY = 0.02  # mild penalty for unjustified large jumps (>3 bid rank steps)
CALL_TIMING_BONUS = 0.03       # extra bonus for calling liar at very high implausibility (>1.5σ)
BLUFF_ZONE_THRESHOLD = 0.40    # bids > 40% of total dice are in bluff zone (high risk)

STRATEGY_TIPS = """
STRATEGY TIPS (Optimal Self-Play Strategy):
- Keep bids minimally stronger than current bid when uncertain (+1 quantity or +1 face).
- Use your own dice + wild 6s to estimate plausible total counts.
- Expected count for any face ≈ total_dice/6 (or total_dice/3 with wild-6).
- Bids claiming >40% of total dice are in the BLUFF ZONE — challenge aggressively.
- Bids claiming >50% of total dice are almost certainly bluffs — call Liar.
- Call Liar when the current bid is statistically implausible given your dice.
- Avoid large overbids (jump >3 bid rank) unless your private dice strongly support it.
- Early game (many dice): bluffs are harder to detect, moderate bids are safer.
- Late game (few dice): bids become more transparent, challenge more freely.
"""

REASONING_TAG_PAIRS = [
    ("think", "think"),
    ("thinking", "thinking"),
    ("reasoning", "reasoning"),
    ("thought", "thought"),
    ("reflection", "reflection"),
]

_ROLLOUT_STATE: dict = {}


# ---------------------------------------------------------------------------
# Fase 3: CFR Table (Counterfactual Regret Minimization)
# ---------------------------------------------------------------------------

class CFRTable:
    """Thread-safe Counterfactual Regret Minimization table for Liar's Dice.

    Tracks regret per (dice_pattern, bid_rank) information set.
    After each episode, regret is updated: if the model bid X and lost,
    R(X) += |loss|; if the model bid X and won, R(X) -= |win| (reward signal).

    Strategy (for shaping):
      σ(a | I) ∝ max(0, R(a))   ← regret-matching

    The table is used to provide a small shaping bonus/penalty based on
    whether the chosen bid aligns with the current cumulative-regret strategy.

    Reference: Zinkevich et al. 2007 (CFR), Brown & Sandholm 2019 (CFR+)
    """

    CFR_SHAPING_SCALE = 0.02   # max shaping magnitude from CFR signal

    def __init__(self) -> None:
        self._lock = Lock()
        # key: (dice_pattern_str, bid_rank:int) → cumulative_regret:float
        self._regret: dict[tuple[str, int], float] = {}
        # key: (dice_pattern_str, bid_rank:int) → cumulative_strategy:float
        self._strategy_sum: dict[tuple[str, int], float] = {}
        self._episode_count: int = 0

    @staticmethod
    def _dice_pattern(own_dice: list[int], total_dice: int = 0) -> str:
        """Canonical dice pattern with game-phase bucket.

        Phase bucketing ensures early / mid / late game strategies are stored
        separately in the regret table, preventing cross-phase contamination.
          E = early  (total_dice >= 8)
          M = mid    (4 <= total_dice < 8)
          L = late   (total_dice < 4)
        """
        pattern = str(tuple(sorted(own_dice))) if own_dice else "()"
        if total_dice >= 8:
            phase = "E"
        elif total_dice >= 4:
            phase = "M"
        else:
            phase = "L"
        return f"{pattern}|{phase}"

    def _get_regrets(self, info_key: str) -> dict[int, float]:
        """Return {bid_rank: regret} for this info_key from the table."""
        result = {}
        prefix = (info_key, 0).__class__  # just to iterate
        with self._lock:
            for (ik, br), r in self._regret.items():
                if ik == info_key:
                    result[br] = r
        return result

    def get_strategy(
        self, own_dice: list[int], legal_bid_ranks: list[int], total_dice: int = 0
    ) -> dict[int, float]:
        """Return regret-matching mixed strategy σ over legal_bid_ranks.

        σ(a) = max(0, R(a)) / Σ max(0, R(a))
        If all regrets ≤ 0, return uniform strategy.
        total_dice: passed for phase-aware info-set key.
        """
        if not own_dice or not legal_bid_ranks:
            n = max(len(legal_bid_ranks), 1)
            return {br: 1.0 / n for br in legal_bid_ranks}

        info_key = self._dice_pattern(own_dice, total_dice)
        regrets = self._get_regrets(info_key)

        positive = {br: max(0.0, regrets.get(br, 0.0)) for br in legal_bid_ranks}
        total = sum(positive.values())

        if total <= 0:
            n = len(legal_bid_ranks)
            return {br: 1.0 / n for br in legal_bid_ranks}
        return {br: v / total for br, v in positive.items()}

    def cfr_shaping(
        self,
        own_dice: list[int],
        chosen_bid_rank: int,
        legal_bid_ranks: list[int],
        total_dice: int = 0,
    ) -> float:
        """Return CFR-based shaping reward for the chosen bid.

        Positive if the chosen bid aligns with the accumulated optimal strategy;
        Negative if it diverges (high-regret action chosen when better options exist).

        Magnitude is always ≤ CFR_SHAPING_SCALE (0.02).
        total_dice: passed for phase-aware info-set key.
        """
        strategy = self.get_strategy(own_dice, legal_bid_ranks, total_dice=total_dice)
        chosen_prob = strategy.get(chosen_bid_rank, 0.0)
        # Centre around 1/N (uniform): positive = above average, negative = below
        uniform_prob = 1.0 / max(len(legal_bid_ranks), 1)
        deviation = chosen_prob - uniform_prob
        # Normalise to [-CFR_SHAPING_SCALE, +CFR_SHAPING_SCALE]
        max_deviation = max(1.0 - uniform_prob, uniform_prob)
        if max_deviation > 0:
            return self.CFR_SHAPING_SCALE * (deviation / max_deviation)
        return 0.0

    def update(
        self,
        own_dice: list[int],
        bid_rank_played: int,
        legal_bid_ranks: list[int],
        episode_reward: float,
        total_dice: int = 0,
    ) -> None:
        """Update regret table after episode completes (regret matching update).
        total_dice: total dice at time of last bid, for phase-aware info-set key.
        """
        if not own_dice or not legal_bid_ranks:
            return
        info_key = self._dice_pattern(own_dice, total_dice)
        played_value = episode_reward

        # Compute strategy BEFORE acquiring lock to avoid self-deadlock
        # (get_strategy → _get_regrets also acquires self._lock)
        strategy = self.get_strategy(own_dice, legal_bid_ranks, total_dice=total_dice)

        with self._lock:
            self._episode_count += 1
            for br in legal_bid_ranks:
                if br == bid_rank_played:
                    self._regret[(info_key, br)] = (
                        self._regret.get((info_key, br), 0.0) - played_value * 0.1
                    )
                else:
                    counterfactual = -played_value * 0.05
                    self._regret[(info_key, br)] = (
                        self._regret.get((info_key, br), 0.0) + counterfactual
                    )
                # Accumulate strategy sum (already computed outside lock)
                self._strategy_sum[(info_key, br)] = (
                    self._strategy_sum.get((info_key, br), 0.0)
                    + strategy.get(br, 0.0)
                )

    def stats(self) -> str:
        with self._lock:
            n_entries = len(self._regret)
            return f"CFRTable: {n_entries} entries, {self._episode_count} episodes"


# ---------------------------------------------------------------------------
# Fase 3: Bayesian Opponent Inference (Liar's Dice)
# ---------------------------------------------------------------------------

class BayesianOpponentInference:
    """Track P(opponent_roll | observed_bids) via sequential Bayesian update.

    Prior: uniform over all 6^n_dice outcomes (e.g. 6^5 = 7776 for 5 dice).
    Update per bid: P(roll | bid) ∝ P(bid | roll) × P(roll)

    P(bid b | roll):
      = 1   if the bid is consistent with the roll
            (i.e., face_count(roll, face, wild6) + own_support ≥ qty)
      = ε   otherwise (bluff is always possible, but down-weighted)

    The posterior gives the expected support for any face value across
    the opponent's dice — more accurate than the Binomial prior alone.
    """

    BLUFF_PROB = 0.15  # default bluff probability (fallback if adaptive is not used)

    @staticmethod
    def _adaptive_bluff_prob(total_dice: int, n_bids_seen: int) -> float:
        """Adaptive bluff probability calibrated to game phase and observation count.

        Late game (fewer dice): bluffing is more common and harder to sustain
            credibly → higher bluff assumption.
        Early game (many dice): harder to be caught bluffing → lower assumption.
        Confidence: more bids observed → posterior sharper → bluff prior matters less.

        Range: [0.08, 0.28]
        """
        phase_factor = max(0.0, 1.0 - (total_dice / 10.0))   # 0.0 early → 1.0 very late
        base_bluff = 0.10 + 0.18 * phase_factor               # [0.10, 0.28]
        # As more bids are observed, posterior is already conditioned → bluff prior matters less
        confidence_decay = min(n_bids_seen / 5.0, 1.0)
        return max(0.08, base_bluff * (1.0 - 0.25 * confidence_decay))

    def __init__(self, n_dice: int = 5, wild_six: bool = True) -> None:
        self.n_dice = n_dice
        self.wild_six = wild_six
        self._all_rolls: list[tuple[int, ...]] = self._enumerate_rolls(n_dice)
        # Log-probability for numerical stability
        n = len(self._all_rolls)
        self._log_probs: list[float] = [-math.log(n)] * n  # uniform prior
        self._roll_index: dict[tuple[int, ...], int] = {
            r: i for i, r in enumerate(self._all_rolls)
        }
        self._bids_observed: int = 0  # count bids seen this episode (for adaptive bluff)

    @staticmethod
    def _enumerate_rolls(n_dice: int) -> list[tuple[int, ...]]:
        """Enumerate all n_dice-tuples of values 1–6."""
        if n_dice <= 0:
            return [()]
        result: list[tuple[int, ...]] = [()]
        for _ in range(n_dice):
            result = [r + (v,) for r in result for v in range(1, 7)]
        return result

    def _face_count(self, roll: tuple[int, ...], face: int) -> int:
        if self.wild_six and face != 6:
            return sum(1 for d in roll if d == face or d == 6)
        return sum(1 for d in roll if d == face)

    def update(self, bid: tuple[int, int], own_support: int, total_dice: int = 10) -> None:
        """Update posterior given opponent's bid (qty, face) and our own support.

        Vectorised with numpy: ~100x faster than pure-Python loop over 7776 rolls.
        total_dice: used for adaptive bluff probability calibration.
        """
        self._bids_observed += 1
        qty, face = bid
        need_from_opp = max(qty - own_support, 0)
        # Adaptive bluff probability: higher in late game, lower as posterior sharpens
        adaptive_bp = self._adaptive_bluff_prob(total_dice, self._bids_observed)
        log_bluff = math.log(adaptive_bp)
        log_support = math.log(1.0 - adaptive_bp)

        # Build likelihood vector using numpy for speed
        try:
            import numpy as np
            # _opp_support_cache: (n_rolls,) array of opp dice count for this face
            if not hasattr(self, '_np_rolls') or self._np_rolls is None:
                self._np_rolls = np.array(self._all_rolls, dtype=np.int8)  # (N, n_dice)
            rolls = self._np_rolls
            if self.wild_six and face != 6:
                opp_support = np.sum((rolls == face) | (rolls == 6), axis=1)
            else:
                opp_support = np.sum(rolls == face, axis=1)
            log_likelihoods = np.where(opp_support >= need_from_opp, log_support, log_bluff)
            log_probs_arr = np.array(self._log_probs) + log_likelihoods
            # Log-sum-exp normalise
            max_lp = log_probs_arr.max()
            log_sum = max_lp + math.log(np.exp(log_probs_arr - max_lp).sum())
            self._log_probs = (log_probs_arr - log_sum).tolist()
        except ImportError:
            # Pure-Python fallback (slow but safe)
            new_log_probs = []
            for i, roll in enumerate(self._all_rolls):
                opp_support = self._face_count(roll, face)
                log_likelihood = log_support if opp_support >= need_from_opp else log_bluff
                new_log_probs.append(self._log_probs[i] + log_likelihood)
            max_lp = max(new_log_probs)
            log_sum = max_lp + math.log(sum(math.exp(lp - max_lp) for lp in new_log_probs))
            self._log_probs = [lp - log_sum for lp in new_log_probs]

    def expected_support(self, face: int) -> float:
        """Expected number of opponent dice showing `face` (+ wild-6 if enabled)."""
        total = 0.0
        for i, roll in enumerate(self._all_rolls):
            prob = math.exp(self._log_probs[i])
            total += prob * self._face_count(roll, face)
        return total

    def bid_posterior_prob(self, bid: tuple[int, int], own_support: int) -> float:
        """P(bid is true | posterior) = probability opponent's roll supports bid."""
        qty, face = bid
        need = max(qty - own_support, 0)
        return sum(
            math.exp(self._log_probs[i])
            for i, roll in enumerate(self._all_rolls)
            if self._face_count(roll, face) >= need
        )

    def call_shaping(
        self,
        bid: tuple[int, int],
        own_support: int,
        call_quality_bonus: float,
        call_quality_penalty: float,
        episode_won: bool,
    ) -> float:
        """Extra shaping for call-liar decisions using Bayesian posterior.

        Replaces or augments the simple implausibility-based call shaping.
        - p_true low + we called (and won) → larger bonus
        - p_true high + we called (and lost) → larger penalty (called too eagerly)
        """
        p_true = self.bid_posterior_prob(bid, own_support)
        if episode_won:
            # Good call: bonus scales inversely with p_true (lower p_true = braver call)
            return call_quality_bonus * (1.0 + (1.0 - p_true))
        else:
            # Bad call: penalty scales with p_true (higher p_true = we shouldn't have called)
            return -call_quality_penalty * (1.0 + p_true)

    def reset(self, n_dice: int | None = None, wild_six: bool | None = None) -> None:
        """Reset to uniform prior (call at start of each episode)."""
        if n_dice is not None:
            self.n_dice = n_dice
        if wild_six is not None:
            self.wild_six = wild_six
        self._all_rolls = self._enumerate_rolls(self.n_dice)
        n = len(self._all_rolls)
        self._log_probs = [-math.log(n)] * n
        self._roll_index = {r: i for i, r in enumerate(self._all_rolls)}
        self._bids_observed = 0  # reset bid counter for adaptive bluff prob

    def context_summary(self, face: int) -> str:
        """Short string for injection into observation."""
        exp = self.expected_support(face)
        return f"[Bayesian] Expected opp dice showing {face}: ~{exp:.1f}"




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
    # Liar's Dice observations already contain structured legal-action blocks.
    return obs_text or ""


class EpisodeTraceLogger:
    """Thread-safe JSONL episode tracer."""

    def __init__(self, trace_dir: str, rank: int):
        self.trace_dir = trace_dir
        self.rank = rank
        self._lock = Lock()
        self.log_path = os.path.join(self.trace_dir, f"liars_dice_episode_traces_rank{rank}.jsonl")
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
    """Progressive turn-limit and MCTS curriculum."""

    def __init__(
        self,
        initial_max_turn: int = 2,
        final_max_turn: int = 20,
        rollouts_per_stage: int = 1280,
        initial_hint_prob: float = 0.0,
        final_hint_prob: float = 0.0,
        warmup_rollouts: int = 128,
        initial_mcts_sims: int = 50,
        final_mcts_sims: int = 300,
        mcts_warmup_optimizer_steps: int = 20,
    ):
        self.initial_max_turn = initial_max_turn
        self.final_max_turn = final_max_turn
        self.rollouts_per_stage = rollouts_per_stage
        self.initial_hint_prob = initial_hint_prob
        self.final_hint_prob = final_hint_prob
        self.warmup_rollouts = warmup_rollouts
        self.initial_mcts_sims = initial_mcts_sims
        self.final_mcts_sims = final_mcts_sims
        self.mcts_warmup_optimizer_steps = mcts_warmup_optimizer_steps
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

    def get_mcts_sims(self, optimizer_step: int | None = None) -> int:
        """Calculate current MCTS simulations based on optimizer step progress.

        Ramps linearly from initial_mcts_sims to final_mcts_sims over
        mcts_warmup_optimizer_steps optimizer steps.
        Tournament eval uses 225 sims; we overshoot to 300 for robustness.
        """
        current_step = 0 if optimizer_step is None else optimizer_step
        if self.mcts_warmup_optimizer_steps <= 0:
            return self.final_mcts_sims
        progress = min(max(current_step, 0) / self.mcts_warmup_optimizer_steps, 1.0)
        return int(
            self.initial_mcts_sims
            + progress * (self.final_mcts_sims - self.initial_mcts_sims)
        )

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


def _extract_bid_tuple(label_or_text: str) -> tuple[int, int] | None:
    if not label_or_text:
        return None
    match = re.search(r"(\d+)\s*-\s*(\d+)", label_or_text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _extract_state_features(observation: str) -> dict:
    dice: list[int] = []
    dice_match = re.search(r"Your dice:\s*\[([^\]]*)\]", observation)
    if dice_match:
        dice_str = dice_match.group(1).strip()
        if dice_str:
            dice = [int(x.strip()) for x in dice_str.split(",") if x.strip().isdigit()]

    total_dice_match = re.search(r"Total dice in game:\s*(\d+)", observation)
    total_dice = int(total_dice_match.group(1)) if total_dice_match else 0

    current_bid_match = re.search(r'Current bid:\s*"([^"]+)"', observation)
    current_bid = _extract_bid_tuple(current_bid_match.group(1)) if current_bid_match else None

    return {
        "own_dice": dice,
        "total_dice": total_dice,
        "current_bid": current_bid,
        "wild_six_enabled": "wild" in observation.lower() and "6" in observation,
    }


def _is_liar_label(label: str) -> bool:
    return "liar" in (label or "").strip().lower()


def _bid_rank(bid: tuple[int, int]) -> int:
    quantity, face = bid
    return quantity * 6 + face


def _count_face_support(own_dice: list[int], target_face: int, wild_six_enabled: bool) -> int:
    if wild_six_enabled and target_face != 6:
        return sum(1 for value in own_dice if value == target_face or value == 6)
    return sum(1 for value in own_dice if value == target_face)


def _score_bid_plausibility(
    state_features: dict,
    bid: tuple[int, int],
    bayes_opp=None,
) -> float:
    """Score bid plausibility using Bayesian posterior (if available) or exact Binomial CDF.

    Path selection:
      - Bayesian path: bayes_opp is available AND _bids_observed > 0 (turn 2+)
        Uses posterior expected opponent support — more accurate than flat Binomial prior
        because the posterior is already conditioned on observed bids.
      - Binomial CDF path: turn 1 (uniform prior, not yet conditioned)
        Uses exact Binomial CDF — more precise than posterior mean when prior is uniform.

    reward = BID_PLAUSIBILITY_BONUS * (2 * p_true - 1)
      p_true = 1.0 → fully plausible  → +BID_PLAUSIBILITY_BONUS (+0.04)
      p_true = 0.5 → uncertain        → 0
      p_true = 0.0 → almost certainly → -BID_PLAUSIBILITY_BONUS (-0.04)
    """
    own_dice = state_features.get("own_dice") or []
    total_dice = int(state_features.get("total_dice") or 0)
    current_bid = state_features.get("current_bid")
    wild_six_enabled = bool(state_features.get("wild_six_enabled"))

    if total_dice <= 0 or not own_dice:
        return 0.0

    quantity, face = bid
    known_support = _count_face_support(own_dice, face, wild_six_enabled)

    # Bayesian path: only when posterior has been updated (at least 1 opp bid observed)
    # → prior is no longer uniform, so posterior mean > Binomial mean in accuracy
    use_bayesian = (
        bayes_opp is not None
        and getattr(bayes_opp, "_bids_observed", 0) > 0
    )

    if use_bayesian:
        # --- Bayesian path: turn 2+ (posterior conditioned on observed bids) ---
        opp_expected = bayes_opp.expected_support(face)
        total_expected = known_support + opp_expected
        deficit = max(quantity - total_expected, 0.0)
        if deficit <= 0.0:
            p_bid_true = 1.0
        else:
            # Linear degradation: deficit approaching total_dice → p_bid_true → 0
            p_bid_true = max(0.0, 1.0 - (deficit / max(float(total_dice), 1.0)))
    else:
        # --- Binomial CDF path: turn 1 or no Bayesian data ---
        # More precise than posterior mean when prior is still uniform
        unknown_dice = max(total_dice - len(own_dice), 0)
        p = 2.0 / 6.0 if (wild_six_enabled and face != 6) else 1.0 / 6.0
        need_from_unknown = max(quantity - known_support, 0)

        if unknown_dice <= 0:
            p_bid_true = 1.0 if need_from_unknown <= 0 else 0.0
        elif need_from_unknown <= 0:
            p_bid_true = 1.0
        else:
            p_bid_true = float(1.0 - _binom.cdf(need_from_unknown - 1, unknown_dice, p))

    reward = BID_PLAUSIBILITY_BONUS * (2.0 * p_bid_true - 1.0)

    # Bid escalation bonus (unchanged)
    if current_bid is not None:
        jump = _bid_rank(bid) - _bid_rank(current_bid)
        if jump <= 2:
            reward += 0.01
        elif jump >= 7:
            if known_support < 2:
                reward -= 0.02
            else:
                reward += 0.01

    return reward


def _score_bid_quality(
    bid: tuple[int, int],
    previous_bid: tuple[int, int] | None,
    state_features: dict,
) -> float:
    """Reward strategically sound bid increments; penalise reckless jumps.

    Phase-aware: safe jump threshold adapts to total_dice remaining.
      Early game  (>=8 dice): safe_jump=3 — many unknowns absorb larger bids
      Mid game  (5–7 dice): safe_jump=2 — standard Nash-optimal increment
      Late game   (<5 dice): safe_jump=1 — every die is transparent; be minimal

    Opening bid (previous_bid=None): penalise if bid exceeds 1.5× expected total.

    Returns a small float in [-BID_AGGRESSIVE_PENALTY, BID_MARGINAL_BONUS].
    """
    own_dice = state_features.get("own_dice") or []
    total_dice = int(state_features.get("total_dice") or 1)
    wild_six = bool(state_features.get("wild_six_enabled"))

    # Opening bid — provide feedback based on support ratio (no prev bid baseline)
    if previous_bid is None:
        qty, face = bid
        known_support = _count_face_support(own_dice, face, wild_six)
        p = 2.0 / 6.0 if (wild_six and face != 6) else 1.0 / 6.0
        expected_total = known_support + max(total_dice - len(own_dice), 0) * p
        if qty > expected_total * 1.5:
            return -BID_AGGRESSIVE_PENALTY * 0.5  # mild penalty for aggressive opening
        return 0.0

    prev_rank = _bid_rank(previous_bid)
    curr_rank = _bid_rank(bid)
    jump = curr_rank - prev_rank

    # Phase-aware safe jump threshold
    if total_dice >= 8:
        safe_jump = 3   # early game: many unknowns → larger jumps tolerable
    elif total_dice >= 5:
        safe_jump = 2   # mid game: standard Nash-optimal
    else:
        safe_jump = 1   # late game: every die visible → minimal increment only

    if jump <= safe_jump:
        return BID_MARGINAL_BONUS  # +0.02

    # Large jump (beyond safe_jump+1) without private support = reckless
    if jump >= safe_jump + 2:
        qty, face = bid
        known_support = _count_face_support(own_dice, face, wild_six)
        expected_unknown = max(total_dice - len(own_dice), 0) * (
            2.0 / 6.0 if (wild_six and face != 6) else 1.0 / 6.0
        )
        expected_total = known_support + expected_unknown
        if qty > expected_total + 1:
            return -BID_AGGRESSIVE_PENALTY  # -0.02

    return 0.0  # moderate jump = neutral


def _adaptive_bluff_threshold(total_dice: int) -> float:
    """Return an adaptive bluff-detection threshold based on total dice remaining.

    Early game (many dice): higher threshold → more tolerant of large bids
        (bids harder to verify with many unknown dice)
    Late game (few dice):   lower threshold → challenge earlier
        (fewer dice make bids more transparent; high ratios are almost certainly bluffs)

    Linear interpolation:
      total_dice = 10 → 0.50  (high tolerance, early game)
      total_dice = 5  → 0.425 (mid game)
      total_dice = 2  → 0.365 (late game, aggressive)
    """
    MAX_STARTING_DICE = 10
    ratio = min(max(total_dice, 0) / MAX_STARTING_DICE, 1.0)
    # Range: [0.35 (late game), 0.50 (early game)]
    return 0.35 + 0.15 * ratio


def _compute_dice_context_summary(state_features: dict, current_bid: tuple[int, int] | None) -> str:
    """Return a short probabilistic context string to inject into each observation.

    Gives the LLM a numerical anchor for evaluating bid plausibility:
      - Expected count per face value (with / without wild-6)
      - Whether the current bid crosses the ADAPTIVE bluff threshold
      - Simple challenge recommendation calibrated to game stage

    Literature basis:
      - Threshold adapts: early game more tolerant, late game more aggressive
        (WiLDCARD, Stanford CFR papers)
    """
    own_dice: list[int] = state_features.get("own_dice") or []
    total_dice = int(state_features.get("total_dice") or 0)
    wild_six = bool(state_features.get("wild_six_enabled"))

    if total_dice == 0:
        return ""

    p = 2.0 / 6.0 if wild_six else 1.0 / 6.0
    expected_per_face = round(total_dice * p, 1)
    threshold = _adaptive_bluff_threshold(total_dice)
    lines = [
        f"[Dice context] Total dice: {total_dice} | Your dice: {own_dice}",
        f"Expected any face across all dice: ~{expected_per_face} "
        f"({'wild-6 active: p=2/6' if wild_six else 'standard: p=1/6'})",
    ]

    if current_bid is not None:
        qty, face = current_bid
        ratio = qty / total_dice
        extreme_threshold = threshold + 0.10  # e.g. 0.60 early / 0.45 late
        if ratio >= extreme_threshold:
            lines.append(
                f"Current bid {qty}×{face}: {ratio:.0%} of total dice "
                f"— EXTREME BLUFF ZONE (threshold {extreme_threshold:.0%}) → Consider calling Liar"
            )
        elif ratio >= threshold:
            lines.append(
                f"Current bid {qty}×{face}: {ratio:.0%} of total dice "
                f"— BLUFF ZONE (threshold {threshold:.0%}) → Challenging is reasonable"
            )
        else:
            lines.append(
                f"Current bid {qty}×{face}: {ratio:.0%} of total dice "
                f"— within normal range (bluff zone starts at {threshold:.0%})"
            )

    return "\n".join(lines)


def _parse_action_id(completion_text: str, legal_action_map: dict[str, str]) -> str:
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

    if "liar" in normalized:
        for action_id, label in legal_action_map.items():
            if _is_liar_label(label):
                return action_id

    bid_tuple = _extract_bid_tuple(cleaned)
    if bid_tuple is not None:
        for action_id, label in legal_action_map.items():
            if _extract_bid_tuple(label) == bid_tuple:
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


def _build_env_pool(server_urls: list[str]) -> list[dict[str, str]]:
    env_pool = []
    init_task_id = GAME_TO_TASK_ID_RANGE[SELECTED_GAME][0]

    for idx, base_url in enumerate(server_urls):
        try:
            print(f"[INIT] Initializing env on server {idx}: {base_url}")
            # Use INITIAL_MCTS_SIMS for probe (faster startup than full MCTS_CONFIG=225)
            payload = {"task_id": init_task_id, "seed": 42, "opponent": "mcts",
                        "mcts_max_simulations": INITIAL_MCTS_SIMS, "mcts_num_rollouts": 1}
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
    final_max_turn = int(os.environ.get("LIARS_DICE_FINAL_MAX_TURN", "20"))
    initial_hint_prob = float(os.environ.get("LIARS_DICE_INITIAL_HINT_PROB", "0.0"))
    final_hint_prob = float(os.environ.get("LIARS_DICE_FINAL_HINT_PROB", "0.0"))

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
        warmup_rollouts=128,
        initial_mcts_sims=INITIAL_MCTS_SIMS,
        final_mcts_sims=FINAL_MCTS_SIMS,
        mcts_warmup_optimizer_steps=int(getattr(trainer.args, "mcts_warmup_optimizer_steps", MCTS_WARMUP_STEPS) or MCTS_WARMUP_STEPS),
    )
    print(
        f"[CURRICULUM] MCTS curriculum: {INITIAL_MCTS_SIMS} -> {FINAL_MCTS_SIMS} "
        f"over {MCTS_WARMUP_STEPS} optimizer steps (tournament eval=225)"
    )
    # Fase 3: CFR Table — shared across all episodes (thread-safe), updated per episode
    _ROLLOUT_STATE["cfr_table"] = CFRTable()
    # Fase 3: BayesianOpponentInference — reset at episode start, updated per opponent bid
    # Created once per rank; the rollout loop calls .reset() at the start of each episode.
    _ROLLOUT_STATE["bayes_opp"] = BayesianOpponentInference(n_dice=5, wild_six=True)
    _ROLLOUT_STATE["initialized"] = True

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


def _reset_environment(env_endpoint: str, game_id: int, timeout: int, mcts_sims: int | None = None) -> tuple[str, str]:
    """Reset environment with optional dynamic MCTS simulation count."""
    if mcts_sims is not None:
        payload = {
            "task_id": game_id,
            "seed": random.randint(0, 2**31 - 1),
            "opponent": "mcts",
            "mcts_max_simulations": mcts_sims,
            "mcts_num_rollouts": 1,
        }
    else:
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
    print(f"[BATCH] Finished: {finished}/{len(list_results)}, AvgReturn: {avg_return:.3f}")


def _bid_history_summary(bid_history: list[dict]) -> str:
    """Format bid history as compact string for injection into observation.

    Each entry: {turn: int, action_id: str, label: str, is_ours: bool}
    Returns e.g.: 'Bid history: T0(us): 2-3 | T1(opp): 3-3 | T2(us): 3-5'
    """
    if not bid_history:
        return ""
    parts = []
    for entry in bid_history[-12:]:  # keep last 12 bids max (avoid prompt bloat)
        turn = entry.get("turn", "?")
        label = entry.get("label", "?")
        who = "us" if entry.get("is_ours") else "opp"
        parts.append(f"T{turn}({who}):{label}")
    return "Bid history: " + " | ".join(parts)


def _get_system_prompt(use_hints: bool, wild_six_enabled: bool = False) -> str:
    wild_note = (
        "- 6s are WILD: they count as any face value (e.g., a 6 supports any bid).\n"
        if wild_six_enabled
        else "- Standard rules: 6s count only as 6 (no wild dice).\n"
    )
    system_prompt = (
        "You are playing liars_dice.\n\n"
        "# Game Rules\n"
        "LIAR'S DICE RULES:\n"
        "- Each player has hidden dice.\n"
        "- Players make increasing bids in format quantity-face.\n"
        "- You can either raise the bid or choose Liar to challenge.\n"
        + wild_note +
        "- Use ONLY legal action IDs from the current Legal Actions block.\n\n"
        "# Output Format\n"
        "Respond with only one legal action ID number.\n"
        "No explanations.\n"
    )
    if use_hints:
        system_prompt += "\n" + STRATEGY_TIPS
    return system_prompt


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
    current_optimizer_step = getattr(getattr(trainer, "state", None), "global_step", 0)
    current_mcts_sims = curriculum.get_mcts_sims(current_optimizer_step)
    print(
        f"[CURRICULUM] Rollout {curriculum.total_rollouts}, step {current_optimizer_step}: "
        f"max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}, mcts_sims={current_mcts_sims}"
    )

    def run_single_prompt(index: int, prompt: str):
        game_id = int(prompt)
        server_idx = (index + rank) % num_servers
        server = env_pool[server_idx]
        env_endpoint = server["base_url"]

        invalid_count = 0
        noop_count = 0
        consecutive_noops = 0       # consecutive identical-obs turns
        done = False
        final_reward = 0.0
        turn_number = 0
        accumulated_shaping_reward = 0.0
        step_records = []
        termination_reason = "unknown"
        last_step_block: dict = {}
        bid_history: list[dict] = []   # tracks all bids this episode
        wild_six_enabled: bool = False  # detected from first observation
        # Fase 3: CFR per-episode tracking
        episode_bid_ranks_played: list[int] = []
        episode_legal_bid_ranks: list[int] = []
        episode_last_bid_total_dice: int = 0  # total_dice at time of last bid (for CFR update)
        cfr_table: CFRTable | None = _ROLLOUT_STATE.get("cfr_table")
        bayes_opp: BayesianOpponentInference | None = _ROLLOUT_STATE.get("bayes_opp")
        if bayes_opp is not None:
            bayes_opp.reset(n_dice=5, wild_six=True)  # fresh prior each episode
        episode_own_dice: list[int] = []  # captured from first observation

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
                mcts_sims=current_mcts_sims,
            )
            print(f"[START] ID:{game_id} server={server_idx} ep={episode_id[:8] if episode_id else '?'} mcts={current_mcts_sims}")
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
        # Detect Wild-6 and capture own dice from the first observation
        state_features_init = _extract_state_features(formatted_observation)
        wild_six_enabled = bool(state_features_init.get("wild_six_enabled"))
        episode_own_dice = list(state_features_init.get("own_dice") or [])   # Fase 3: for CFR/Bayes
        if bayes_opp is not None and episode_own_dice:
            bayes_opp.reset(n_dice=5, wild_six=wild_six_enabled)  # re-reset with correct wild flag
        messages = [
            {"role": "system", "content": _get_system_prompt(use_hints=use_hints, wild_six_enabled=wild_six_enabled)},
            {"role": "user", "content": formatted_observation},
        ]

        while not done and turn_number < current_max_turn:
            observation_before_action = formatted_observation
            legal_action_map = _extract_legal_action_map(observation_before_action)
            state_features = _extract_state_features(observation_before_action)
            # Override wild_six_enabled from episode-level detection.
            # Step-level observations don't contain "wild" text (only /reset does),
            # so re-detecting from observation would incorrectly return False on turn 2+.
            state_features["wild_six_enabled"] = wild_six_enabled

            # Fase 3: Update Bayesian posterior with opponent's bid (current_bid = opp's last bid)
            # When it's our turn, the current_bid in state was placed by the opponent.
            if bayes_opp is not None and episode_own_dice:
                opp_bid = state_features.get("current_bid")
                if opp_bid is not None:
                    opp_own_support = _count_face_support(
                        episode_own_dice, opp_bid[1], wild_six_enabled
                    )
                    bayes_opp.update(
                        opp_bid,
                        opp_own_support,
                        total_dice=int(state_features.get("total_dice") or 10),
                    )

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
            action_label = legal_action_map.get(action_to_send, "")
            liar_action = _is_liar_label(action_label)
            parsed_bid = _extract_bid_tuple(action_label)

            if action_to_send not in legal_action_map:
                invalid_count += 1
                accumulated_shaping_reward -= INVALID_ACTION_PENALTY
                action_to_send = sorted(legal_action_map.keys(), key=lambda x: int(x))[0]
                action_label = legal_action_map.get(action_to_send, "")
                liar_action = _is_liar_label(action_label)
                parsed_bid = _extract_bid_tuple(action_label)

            # --- Record bid in history ---
            if parsed_bid is not None:
                bid_history.append({
                    "turn": turn_number,
                    "action_id": action_to_send,
                    "label": action_label,
                    "is_ours": True,
                })
            elif liar_action:
                bid_history.append({
                    "turn": turn_number,
                    "action_id": action_to_send,
                    "label": "LIAR",
                    "is_ours": True,
                })

            bid_shaping = 0.0
            call_shaping = 0.0
            if parsed_bid is not None:
                # Fix #5: pass bayes_opp for Bayesian-informed plausibility at turn 2+
                bid_shaping = _score_bid_plausibility(state_features, parsed_bid, bayes_opp=bayes_opp)
                # Self-play RL: also score *quality* of bid increment (marginal vs reckless)
                previous_bid = state_features.get("current_bid")  # bid that was in play before our action
                bid_quality = _score_bid_quality(parsed_bid, previous_bid, state_features)
                bid_shaping += bid_quality
                # Fase 3: CFR shaping — align bid choice with accumulated optimal strategy
                if cfr_table is not None and episode_own_dice:
                    br = _bid_rank(parsed_bid)
                    legal_brs = [
                        _bid_rank(_extract_bid_tuple(lbl))
                        for lbl in legal_action_map.values()
                        if _extract_bid_tuple(lbl) is not None
                    ]
                    if legal_brs:
                        # Fix #1: pass total_dice for phase-aware CFR shaping
                        _td = int(state_features.get("total_dice") or 0)
                        bid_shaping += cfr_table.cfr_shaping(
                            episode_own_dice, br, legal_brs, total_dice=_td
                        )
                        episode_bid_ranks_played.append(br)
                        episode_last_bid_total_dice = _td  # track for CFR update
                        episode_legal_bid_ranks = list(set(episode_legal_bid_ranks + legal_brs))
                accumulated_shaping_reward += bid_shaping

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

            invalid_or_noop = (
                "Invalid" in formatted_observation
                or "Nothing happens" in formatted_observation
                or action_to_send not in legal_action_map
            )
            if invalid_or_noop:
                invalid_count += 1
                accumulated_shaping_reward -= INVALID_ACTION_PENALTY

            if formatted_observation == observation_before_action:
                noop_count += 1
                consecutive_noops += 1
                accumulated_shaping_reward -= NOOP_PENALTY
                # Escalating penalty for persistent noops (e.g., stuck in loop)
                if consecutive_noops >= 3:
                    accumulated_shaping_reward -= NOOP_PENALTY * (consecutive_noops - 2)
            else:
                consecutive_noops = 0  # reset on state change

            if done:
                final_reward = _extract_terminal_reward(last_step_block, formatted_observation)
                if liar_action and state_features.get("current_bid") is not None:
                    cb = state_features["current_bid"]
                    cb_quantity, cb_face = cb
                    cb_own_dice = state_features.get("own_dice") or []
                    cb_known = _count_face_support(
                        cb_own_dice,
                        cb_face,
                        bool(state_features.get("wild_six_enabled")),
                    )
                    cb_total_dice = int(state_features.get("total_dice") or 1)
                    # Fase 3: Use Bayesian posterior for call_shaping (more accurate than implausibility)
                    if bayes_opp is not None:
                        call_shaping = bayes_opp.call_shaping(
                            bid=cb,
                            own_support=cb_known,
                            call_quality_bonus=CALL_QUALITY_BONUS,
                            call_quality_penalty=CALL_QUALITY_PENALTY,
                            episode_won=(final_reward > 0),
                        )
                        # Symmetry with fallback path: extra bonus for calling at very
                        # high implausibility (brave call when posterior strongly disagrees)
                        if final_reward > 0:
                            p_true_timing = bayes_opp.bid_posterior_prob(cb, cb_known)
                            if p_true_timing < 0.30:
                                call_shaping += CALL_TIMING_BONUS
                    else:
                        # Fallback: original implausibility-based shaping
                        cb_unknown = max(cb_total_dice - len(cb_own_dice), 0)
                        cb_p = (
                            2.0 / 6.0
                            if (bool(state_features.get("wild_six_enabled")) and cb_face != 6)
                            else 1.0 / 6.0
                        )
                        cb_expected = cb_known + cb_unknown * cb_p
                        implausibility = (cb_quantity - cb_expected) / max(cb_total_dice, 1)
                        if final_reward > 0:
                            call_shaping = CALL_QUALITY_BONUS * (1.0 + _clamp(implausibility, 0.0, 2.0))
                            cb_std = math.sqrt(cb_total_dice * cb_p * (1.0 - cb_p))
                            if cb_std > 0:
                                z_score = (cb_quantity - (cb_total_dice * cb_p)) / cb_std
                                if z_score > 1.5:
                                    call_shaping += CALL_TIMING_BONUS
                        elif final_reward < 0:
                            call_shaping = -CALL_QUALITY_PENALTY * (1.0 + _clamp(-implausibility, 0.0, 1.0))
                    accumulated_shaping_reward += call_shaping
                elif liar_action:
                    if final_reward > 0:
                        call_shaping = CALL_QUALITY_BONUS
                    elif final_reward < 0:
                        call_shaping = -CALL_QUALITY_PENALTY
                    accumulated_shaping_reward += call_shaping
                termination_reason = "done"
            else:
                # Inject bid history + probabilistic dice context into next user message
                next_state_features = _extract_state_features(formatted_observation)
                next_bid = next_state_features.get("current_bid")
                history_summary = _bid_history_summary(bid_history)
                dice_context = _compute_dice_context_summary(next_state_features, next_bid)
                addendum_parts = [p for p in [history_summary, dice_context] if p]
                obs_with_context = (
                    formatted_observation + "\n\n" + "\n".join(addendum_parts)
                    if addendum_parts else formatted_observation
                )
                messages.append({"role": "user", "content": obs_with_context})

            step_records.append(
                {
                    "turn": turn_number,
                    "assistant_text": trace_logger.clip_text(completion_text) if trace_logger else completion_text,
                    "parsed_action": action_to_send,
                    "action_label": action_label,
                    "observation_before_action": (
                        trace_logger.clip_text(observation_before_action)
                        if trace_logger
                        else observation_before_action
                    ),
                    "observation_after_action": (
                        trace_logger.clip_text(formatted_observation) if trace_logger else formatted_observation
                    ),
                    "step_reward": float(step_reward),
                    "bid_shaping": float(bid_shaping),
                    "call_shaping": float(call_shaping),
                    "done": bool(done),
                    "invalid_or_noop": invalid_or_noop,
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
        train_reward = final_reward + clipped_shaping

        # Fase 3: Update CFR table with episode outcome
        if cfr_table is not None and episode_own_dice and episode_bid_ranks_played and episode_legal_bid_ranks:
            last_bid_rank = episode_bid_ranks_played[-1]
            cfr_table.update(
                own_dice=episode_own_dice,
                bid_rank_played=last_bid_rank,
                legal_bid_ranks=episode_legal_bid_ranks,
                episode_reward=train_reward,
                total_dice=episode_last_bid_total_dice,  # Fix #1: phase-aware update
            )

        print(
            f"[ID:{game_id} Hints:{int(use_hints)} Done:{int(done)} T:{turn_number:2d} "
            f"Ret:{train_reward:+.2f} EnvR:{final_reward:+.1f} "
            f"Shape:{clipped_shaping:+.3f} Inv:{invalid_count}"
        )

        if trace_logger and trace_logger.should_log():
            trace_logger.log_episode(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "game_id": game_id,
                    "episode_id": episode_id,
                    "environment": "liars_dice",
                    "status": "completed" if done else "truncated",
                    "termination_reason": termination_reason,
                    "turns": turn_number,
                    "final_reward": float(final_reward),
                    "raw_shaping_reward": float(accumulated_shaping_reward),
                    "clipped_shaping_reward": float(clipped_shaping),
                    "train_reward": float(train_reward),
                    "invalid_count": invalid_count,
                    "noop_count": noop_count,
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
    del max_turns  # Curriculum controls effective horizon.
    return _rollout_parallelized_curriculum(prompts=prompts, trainer=trainer, include_action_mask=False)


def rollout_full_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    del max_turns  # Curriculum controls effective horizon.
    return _rollout_parallelized_curriculum(prompts=prompts, trainer=trainer, include_action_mask=True)


def rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)
