"""
Arena-style Elo rating system using cost-sensitive scoring.

Computes Elo ratings for models based on pairwise comparisons
derived from per-highlight scores. Scoring uses a confusion matrix
calibration to compute posterior expected utilities.
"""

from dataclasses import dataclass, field
from functools import cached_property
from typing import TypedDict

import numpy as np


# =============================================================================
# Type Definitions
# =============================================================================


class PairwiseOutcome(TypedDict):
    """A single pairwise comparison outcome."""

    highlight_id: int
    model_a: str
    model_b: str
    outcome: float  # 1.0 = a wins, 0.0 = b wins, 0.5 = tie


class HighlightMetrics(TypedDict):
    """Metrics for a single highlight."""

    score: float
    benefit: float
    cost: float
    num_prompts: int


class WinLossTie(TypedDict):
    """Win/loss/tie record for a model."""

    wins: int
    losses: int
    ties: int


class ArenaResults(TypedDict):
    """Complete arena evaluation results."""

    ratings: dict[str, float]  # model -> Elo rating
    win_loss_tie: dict[str, WinLossTie]  # model -> {wins, losses, ties}
    n_highlights: int
    n_runs: int


# =============================================================================
# Scoring Policy
# =============================================================================


@dataclass
class ScoringPolicy:
    """Defines how to score model outputs based on judge tiers.

    Attributes:
        name: Identifier for this scoring policy
        confusion_matrix: 4x4 matrix where rows=human truth, cols=judge prediction
        utilities: Dict mapping tier (T0-T3) to utility value
        tiers: List of tier names in order (default: ["T0", "T1", "T2", "T3"])
    """

    name: str
    confusion_matrix: np.ndarray  # shape (4, 4)
    utilities: dict[str, float]  # {"T0": -3.0, "T1": -3.0, "T2": +1.5, "T3": +2.0}
    tiers: list[str] = field(default_factory=lambda: ["T0", "T1", "T2", "T3"])

    def __post_init__(self) -> None:
        """Validate the confusion matrix shape."""
        if self.confusion_matrix.shape != (4, 4):
            raise ValueError(f"Confusion matrix must be 4x4, got {self.confusion_matrix.shape}")

    @cached_property
    def _judge_to_human_posterior(self) -> np.ndarray:
        """P(human tier | judge tier) - posterior distribution over human tiers."""
        # Each column j represents: given judge said tier j, what's the distribution of human tiers?
        return self.confusion_matrix / self.confusion_matrix.sum(axis=0, keepdims=True)

    @cached_property
    def posterior_utilities(self) -> dict[str, float]:
        """E[utility(human) | judge tier] for each judge tier.

        Computes the expected utility when the human would review a prompt,
        given what the judge predicted. This debiases the judge's output
        using the confusion matrix.
        """
        utility_vector = np.array([self.utilities[t] for t in self.tiers])
        result = {}
        for j, judge_tier in enumerate(self.tiers):
            expected_util = float(np.dot(self._judge_to_human_posterior[:, j], utility_vector))
            result[judge_tier] = expected_util
        return result

    @cached_property
    def junk_probabilities(self) -> dict[str, float]:
        """P(human is T0 or T1 | judge tier) for each judge tier.

        This is the probability that a prompt judged at a given tier
        would actually be considered "junk" (T0 or T1) by a human.
        """
        result = {}
        for j, judge_tier in enumerate(self.tiers):
            # P(T0 | judge) + P(T1 | judge)
            prob_junk = float(self._judge_to_human_posterior[0, j] + self._judge_to_human_posterior[1, j])
            result[judge_tier] = prob_junk
        return result


# Default scoring policy which adjusts the
CALIBRATED_JUDGE = ScoringPolicy(
    name="default_calibration",
    confusion_matrix=np.array(
        [
            [44, 28, 1, 1],  # Human T0
            [9, 53, 20, 5],  # Human T1
            [2, 9, 34, 28],  # Human T2
            [5, 13, 25, 9],  # Human T3
        ]
    ),
    utilities={"T0": -1.0, "T1": -3.0, "T2": +1.5, "T3": +2.0},
)


def score_highlight(
    generated_prompts: list[dict],
    policy: ScoringPolicy,
    lambda_junk: float = 1.0,
) -> HighlightMetrics:
    """Score a highlight using the scoring policy.

    For each prompt, looks up posterior_utility and junk_probability from policy.
    - Benefit (B) = max posterior utility across prompts (best card you could show)
    - Cost (C) = sum of junk probabilities across all prompts
    - Score = B - lambda_junk * C

    Args:
        generated_prompts: List of prompt dicts with 'judged_tier' field
        policy: ScoringPolicy with confusion matrix and utilities
        lambda_junk: Penalty weight for junk prompts (default 1.0)

    Returns:
        HighlightMetrics with score, benefit, cost, and num_prompts
    """
    if len(generated_prompts) == 0:
        return HighlightMetrics(score=0.0, benefit=0.0, cost=0.0, num_prompts=0)

    utilities = []
    junk_probs = []

    for prompt in generated_prompts:
        judge_tier = prompt["judged_tier"]
        utilities.append(policy.posterior_utilities[judge_tier])
        junk_probs.append(policy.junk_probabilities[judge_tier])

    benefit = max(utilities)
    cost = sum(junk_probs)
    score = benefit - lambda_junk * cost

    return HighlightMetrics(
        score=score,
        benefit=benefit,
        cost=cost,
        num_prompts=len(generated_prompts),
    )


def expected_score(r_a: float, r_b: float, scale: float = 400.0) -> float:
    """Expected score for player A vs B.

    Returns the probability that A beats B given their ratings.
    A 400-point difference corresponds to a 10:1 expected win ratio.

    Args:
        r_a: Rating of player A
        r_b: Rating of player B
        scale: Rating scale factor (default 400 for standard Elo)

    Returns:
        Expected score for A (probability A beats B)
    """
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / scale))


def update_elo(
    r_a: float,
    r_b: float,
    s_a: float,
    k: float = 32.0,
) -> tuple[float, float]:
    """Update Elo ratings for a single match.

    Args:
        r_a: Rating of player A
        r_b: Rating of player B
        s_a: Score for A (1.0 win, 0.5 draw, 0.0 loss)
        k: K-factor controlling rating volatility

    Returns:
        Tuple of (new_rating_a, new_rating_b)
    """
    e_a = expected_score(r_a, r_b)
    new_r_a = r_a + k * (s_a - e_a)
    new_r_b = r_b + k * ((1.0 - s_a) - (1.0 - e_a))
    return new_r_a, new_r_b
