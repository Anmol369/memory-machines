#!/usr/bin/env python3
"""
Calculate expected utilities accounting for judge error.

Uses Bayesian calibration to compute posterior expected utilities given
a confusion matrix between human tiers and judge predictions.

Utilities reflect the asymmetric costs:
- T0 = −1.0: Obviously wrong, rejected at a glance
- T1 = −3.0: Looks plausible but doesn't support stable recall (worst failure mode)
- T2 = +1.5: Good prompts that contribute to usable deck
- T3 = +2.0: Excellent prompts

Usage:
    python -m memory_machines.arena.judge_error path/to/classified_tasks.jsonl
    python -m memory_machines.arena.judge_error path/to/classified_tasks.jsonl --run-id claude-sonnet-4-5-grounded
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Utility values reflecting review cost asymmetry
DEFAULT_UTILITIES = {"T0": -1.0, "T1": -3.0, "T2": +1.5, "T3": +2.0}
TIER_ORDER = ["T0", "T1", "T2", "T3"]


def build_confusion_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build 4x4 confusion matrix where rows=human truth, cols=judge prediction."""
    y_true = df["expected_tier"].to_numpy()
    y_pred = df["prediction_tier"].to_numpy()
    return confusion_matrix(y_true, y_pred, labels=TIER_ORDER)


def compute_posterior_utilities(
    cm: np.ndarray,
    utilities: dict[str, float] = DEFAULT_UTILITIES,
) -> dict[str, float]:
    """
    Compute E[utility(human) | judge tier] for each judge prediction.

    Uses Bayes' theorem:
        P(human_tier | judge_tier) = cm[human, judge] / sum(cm[:, judge])

    Then:
        E[utility | judge_tier] = sum_i P(human_i | judge) * utility(human_i)

    Args:
        cm: 4x4 confusion matrix (rows=human truth, cols=judge prediction)
        utilities: Dict mapping tier to utility value

    Returns:
        Dict mapping each judge tier to its posterior expected utility
    """
    # Avoid division by zero for columns with no predictions
    col_sums = cm.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0, 1, col_sums)

    # P(human tier | judge tier)
    posterior = cm / col_sums

    # Utility vector in tier order
    utility_vec = np.array([utilities[t] for t in TIER_ORDER])

    # E[utility | judge tier j] = sum_i P(human_i | judge_j) * utility_i
    result = {}
    for j, tier in enumerate(TIER_ORDER):
        expected_util = float(np.dot(posterior[:, j], utility_vec))
        result[tier] = expected_util

    return result


def compute_junk_probabilities(cm: np.ndarray) -> dict[str, float]:
    """
    Compute P(human is T0 or T1 | judge tier) for each judge prediction.

    This is the probability that a prompt judged at a given tier
    would actually be considered "junk" (T0 or T1) by a human.
    """
    col_sums = cm.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0, 1, col_sums)
    posterior = cm / col_sums

    result = {}
    for j, tier in enumerate(TIER_ORDER):
        prob_junk = float(posterior[0, j] + posterior[1, j])  # P(T0|j) + P(T1|j)
        result[tier] = prob_junk

    return result


def print_results(
    cm: np.ndarray,
    posterior_utils: dict[str, float],
    junk_probs: dict[str, float],
    utilities: dict[str, float],
    run_id: str | None = None,
):
    """Print formatted results."""
    title = f"Judge Error Analysis" + (f" ({run_id})" if run_id else "")
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)

    print("\nConfusion Matrix (rows=human truth, cols=judge prediction):")
    print(f"{'':>8}", end="")
    for t in TIER_ORDER:
        print(f"{t:>8}", end="")
    print()
    for i, t in enumerate(TIER_ORDER):
        print(f"{t:>8}", end="")
        for j in range(4):
            print(f"{cm[i, j]:>8}", end="")
        print()

    print(f"\nRaw Utilities: {utilities}")

    print("\nPosterior Expected Utilities (E[utility | judge prediction]):")
    for tier in TIER_ORDER:
        raw = utilities[tier]
        posterior = posterior_utils[tier]
        diff = posterior - raw
        print(f"  {tier}: {posterior:+.3f} (raw: {raw:+.1f}, Δ={diff:+.3f})")

    print("\nJunk Probability (P(T0 or T1 | judge prediction)):")
    for tier in TIER_ORDER:
        prob = junk_probs[tier]
        print(f"  {tier}: {prob:.1%}")

    # Overall metrics
    total = cm.sum()
    correct = np.trace(cm)
    accuracy = correct / total if total > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.1%} ({correct}/{total})")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate expected utilities accounting for judge error"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to classified tasks JSONL file",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Filter to specific run_id (e.g., 'claude-sonnet-4-5-grounded')",
    )
    parser.add_argument(
        "--utilities",
        type=str,
        default=None,
        help="Custom utilities as JSON (e.g., '{\"T0\": -1, \"T1\": -3, \"T2\": 1.5, \"T3\": 2}')",
    )

    args = parser.parse_args()

    # Load data
    df = pd.read_json(args.input_file, lines=True)
    print(f"Loaded {len(df)} records from {args.input_file}")

    # Filter by run_id if specified
    if args.run_id:
        df = df[df["run_id"] == args.run_id]
        print(f"Filtered to {len(df)} records for run_id='{args.run_id}'")

    if len(df) == 0:
        print("No records found. Available run_ids:")
        all_df = pd.read_json(args.input_file, lines=True)
        for run_id in all_df["run_id"].unique():
            print(f"  - {run_id}")
        return

    # Parse utilities if provided
    utilities = DEFAULT_UTILITIES
    if args.utilities:
        import json

        utilities = json.loads(args.utilities)

    # Build confusion matrix and compute metrics
    cm = build_confusion_matrix(df)
    posterior_utils = compute_posterior_utilities(cm, utilities)
    junk_probs = compute_junk_probabilities(cm)

    # Print results
    print_results(cm, posterior_utils, junk_probs, utilities, args.run_id)


if __name__ == "__main__":
    main()
