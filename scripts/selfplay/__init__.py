"""Self-play data layer (Grow phase) for the iterative ReST pipeline.

This package generates training data for env-reposado by *self-play rejection
sampling*: the policy currently being trained plays games against the env-server
MCTS opponent, and only good trajectories survive into the supervised set. It is
deliberately NOT an expert-heuristic data generator.

Pipeline shape (one ReST iteration):

    rollout_collector  ->  trajectory_filter  ->  cot_synthesizer  ->  SFT samples
       (Grow)               (hard-filter)          (STaR rationalize)

References:
  - ReST: Gulcehre et al. 2023, arXiv 2308.08998
  - ReST^EM: Singh et al. 2023, arXiv 2312.06585
  - STaR: Zelikman et al. 2022, arXiv 2203.14465
"""

from .rollout_collector import (
    Episode,
    Turn,
    collect_episodes,
    play_episode,
    parse_thought_action,
)

__all__ = [
    "Episode",
    "Turn",
    "collect_episodes",
    "play_episode",
    "parse_thought_action",
]
