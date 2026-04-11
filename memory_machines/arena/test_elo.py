import numpy as np
import pytest

from memory_machines.arena.elo import (
    CALIBRATED_JUDGE,
    ScoringPolicy,
    score_highlight,
)


class TestScoringPolicy:
    """Tests for ScoringPolicy dataclass."""

    def test_posterior_utilities_computed_correctly(self) -> None:
        """Verify E[utility|judge tier] computation matches manual calculation."""
        # Use a simple confusion matrix for easy verification
        # Perfect judge: diagonal matrix
        policy = ScoringPolicy(
            name="perfect",
            confusion_matrix=np.eye(4),
            utilities={"T0": -3.0, "T1": -3.0, "T2": +1.5, "T3": +2.0},
        )

        # With perfect judge, posterior utility = raw utility
        assert policy.posterior_utilities["T0"] == pytest.approx(-3.0)
        assert policy.posterior_utilities["T1"] == pytest.approx(-3.0)
        assert policy.posterior_utilities["T2"] == pytest.approx(1.5)
        assert policy.posterior_utilities["T3"] == pytest.approx(2.0)

    def test_junk_probabilities_computed_correctly(self) -> None:
        """Verify P(T0/T1|judge tier) computation."""
        # Perfect judge: diagonal matrix
        policy = ScoringPolicy(
            name="perfect",
            confusion_matrix=np.eye(4),
            utilities={"T0": -3.0, "T1": -3.0, "T2": +1.5, "T3": +2.0},
        )

        # With perfect judge, junk probability is 1 for T0/T1, 0 for T2/T3
        assert policy.junk_probabilities["T0"] == pytest.approx(1.0)
        assert policy.junk_probabilities["T1"] == pytest.approx(1.0)
        assert policy.junk_probabilities["T2"] == pytest.approx(0.0)
        assert policy.junk_probabilities["T3"] == pytest.approx(0.0)

    def test_custom_policy(self) -> None:
        """Verify custom confusion matrix works."""
        # 50% confusion between T2 and T3
        confusion = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0.5, 0.5],
                [0, 0, 0.5, 0.5],
            ]
        )
        policy = ScoringPolicy(
            name="custom",
            confusion_matrix=confusion,
            utilities={"T0": 0.0, "T1": 0.0, "T2": 1.0, "T3": 2.0},
        )

        # When judge says T2, 50% chance human says T2, 50% T3
        # E[utility|T2] = 0.5 * 1.0 + 0.5 * 2.0 = 1.5
        assert policy.posterior_utilities["T2"] == pytest.approx(1.5)
        assert policy.posterior_utilities["T3"] == pytest.approx(1.5)


# =============================================================================
# TestScoreHighlight
# =============================================================================


class TestScoreHighlight:
    """Tests for score_highlight function."""

    def test_all_t3_high_score(self) -> None:
        """Best tier gives high benefit."""
        # Use perfect judge for predictable behavior
        policy = ScoringPolicy(
            name="perfect",
            confusion_matrix=np.eye(4),
            utilities={"T0": -3.0, "T1": -3.0, "T2": +1.5, "T3": +2.0},
        )
        prompts = [{"judged_tier": "T3"}, {"judged_tier": "T3"}]
        metrics = score_highlight(prompts, policy)

        assert metrics["benefit"] == pytest.approx(2.0)
        assert metrics["cost"] == pytest.approx(0.0)  # No junk
        assert metrics["score"] == pytest.approx(2.0)  # benefit - 0

    def test_all_t0_low_score(self) -> None:
        """Worst tier gives low benefit + high junk cost."""
        policy = ScoringPolicy(
            name="perfect",
            confusion_matrix=np.eye(4),
            utilities={"T0": -3.0, "T1": -3.0, "T2": +1.5, "T3": +2.0},
        )
        prompts = [{"judged_tier": "T0"}, {"judged_tier": "T0"}]
        metrics = score_highlight(prompts, policy)

        assert metrics["benefit"] == pytest.approx(-3.0)
        assert metrics["cost"] == pytest.approx(2.0)  # 2 junk prompts
        assert metrics["score"] == pytest.approx(-5.0)  # -3.0 - 1.0 * 2.0

    def test_empty_returns_zero(self) -> None:
        """Empty list should return zeros."""
        metrics = score_highlight([], CALIBRATED_JUDGE)
        assert metrics["score"] == 0.0
        assert metrics["benefit"] == 0.0
        assert metrics["cost"] == 0.0
        assert metrics["num_prompts"] == 0

    def test_different_policies_produce_different_scores(self) -> None:
        """Verify policy abstraction works - different policies give different scores."""
        prompts = [{"judged_tier": "T2"}]

        policy_a = ScoringPolicy(
            name="policy_a",
            confusion_matrix=np.eye(4),
            utilities={"T0": 0.0, "T1": 0.0, "T2": 1.0, "T3": 2.0},
        )
        policy_b = ScoringPolicy(
            name="policy_b",
            confusion_matrix=np.eye(4),
            utilities={"T0": 0.0, "T1": 0.0, "T2": 10.0, "T3": 20.0},
        )

        score_a = score_highlight(prompts, policy_a)["score"]
        score_b = score_highlight(prompts, policy_b)["score"]

        assert score_a != score_b
        assert score_b == pytest.approx(10.0)
