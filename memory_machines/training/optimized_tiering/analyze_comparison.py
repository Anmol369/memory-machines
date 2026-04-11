#!/usr/bin/env python3
"""
Analyze and visualize DSPy vs tiering classifier comparison results.

Scans the memory_machines/training/optimized_tiering/results folder for comparison JSONL files
and generates side-by-side confusion matrices.

Usage:
    python -m memory_machines.training.optimized_tiering.analyze_comparison                    # Use default paths
    python -m memory_machines.training.optimized_tiering.analyze_comparison --out-dir ./custom # Custom output
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def visualize_confusion_matrices(df: pd.DataFrame, out_dir: str):
    """
    Visualize confusion matrices for DSPy vs tiering classifier comparison.

    Creates one graph per tiering model with two subplots (DSPy and tiering classifier).

    Args:
        df: DataFrame containing classification results with columns:
            - run_id: The run identifier (e.g., "dspy-grounded", "claude-sonnet-4-5-grounded")
            - prediction_tier: The predicted tier (T0, T1, T2, T3)
            - expected_tier: The actual tier (T0, T1, T2, T3)
        out_dir: Directory to save the visualizations
    """
    # Define the order of tiers for consistent confusion matrices
    tier_order = ["T0", "T1", "T2", "T3"]

    # Separate DSPy and tiering results
    dspy_df = df[df["run_id"] == "dspy-grounded"]
    tiering_df = df[df["run_id"] != "dspy-grounded"]

    if len(dspy_df) == 0:
        print("No DSPy results found")
        return

    if len(tiering_df) == 0:
        print("No tiering classifier results found")
        return

    # Get unique tiering models
    tiering_models = tiering_df["run_id"].unique()

    os.makedirs(out_dir, exist_ok=True)

    for tiering_model in tiering_models:
        tiering_model_df = tiering_df[tiering_df["run_id"] == tiering_model]

        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot DSPy confusion matrix
        y_true_dspy = dspy_df["expected_tier"].to_numpy()  # type: ignore
        y_pred_dspy = dspy_df["prediction_tier"].to_numpy()  # type: ignore
        cm_dspy = confusion_matrix(y_true_dspy, y_pred_dspy, labels=tier_order)

        correct_dspy = int((y_true_dspy == y_pred_dspy).sum())  # type: ignore
        total_dspy = len(y_true_dspy)
        accuracy_dspy = correct_dspy / total_dspy if total_dspy > 0 else 0

        sns.heatmap(
            cm_dspy,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=tier_order,
            yticklabels=tier_order,
            ax=axes[0],
            cbar=True,
            vmin=0,
        )

        axes[0].set_xlabel("Predicted", fontsize=10)
        axes[0].set_ylabel("Expected", fontsize=10)
        axes[0].set_title(
            f"DSPy Grounded\nAccuracy: {accuracy_dspy:.2%} ({correct_dspy}/{total_dspy})",
            fontsize=12,
            fontweight="bold",
        )
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right", fontsize=9)
        axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, fontsize=9)

        # Plot tiering classifier confusion matrix
        y_true_tiering = tiering_model_df["expected_tier"].to_numpy()  # type: ignore
        y_pred_tiering = tiering_model_df["prediction_tier"].to_numpy()  # type: ignore
        cm_tiering = confusion_matrix(y_true_tiering, y_pred_tiering, labels=tier_order)

        correct_tiering = int((y_true_tiering == y_pred_tiering).sum())  # type: ignore
        total_tiering = len(y_true_tiering)
        accuracy_tiering = correct_tiering / total_tiering if total_tiering > 0 else 0

        sns.heatmap(
            cm_tiering,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=tier_order,
            yticklabels=tier_order,
            ax=axes[1],
            cbar=True,
            vmin=0,
        )

        axes[1].set_xlabel("Predicted", fontsize=10)
        axes[1].set_ylabel("Expected", fontsize=10)
        tiering_label = tiering_model.replace("-grounded", "")
        axes[1].set_title(
            f"{tiering_label} Grounded\nAccuracy: {accuracy_tiering:.2%} ({correct_tiering}/{total_tiering})",
            fontsize=12,
            fontweight="bold",
        )
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right", fontsize=9)
        axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0, fontsize=9)

        # Add overall title
        fig.suptitle(
            f"DSPy vs {tiering_label} Comparison",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        safe_model_name = tiering_label.replace("/", "_").replace(":", "_")
        output_path = os.path.join(out_dir, f"comparison_{safe_model_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Comparison visualization saved to {output_path}")


def main(results_dir: str, out_dir: str):
    """Scan results directory and generate visualizations for comparison results."""
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    # Find all .jsonl files in the results directory
    result_files = [f for f in os.listdir(results_dir) if f.endswith(".jsonl") and f.startswith("comparison")]

    if not result_files:
        print(f"No comparison result files found in {results_dir}")
        return

    print(f"Found {len(result_files)} result file(s) in {results_dir}:")
    for f in result_files:
        print(f"  - {f}")

    # Process each result file
    all_dfs = []
    for result_file in result_files:
        results_path = os.path.join(results_dir, result_file)
        try:
            df = pd.read_json(results_path, lines=True)
            print(f"\nLoaded {len(df)} results from {result_file}")

            # Show breakdown by run_id
            if "run_id" in df.columns:
                print("Run ID counts:")
                print(df["run_id"].value_counts())

            all_dfs.append(df)
        except Exception as e:
            print(f"Error loading {result_file}: {e}")

    if not all_dfs:
        print("No valid result files could be loaded")
        return

    # Combine all results
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\n{'=' * 60}")
    print(f"Combined total: {len(combined_df)} results")
    print(f"{'=' * 60}")

    # Generate visualization
    visualize_confusion_matrices(combined_df, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze DSPy vs tiering classifier comparison results")

    # Get default paths relative to this script
    repo_root = Path(__file__).resolve().parents[3]
    default_results_dir = repo_root / "memory_machines" / "training" / "optimized_tiering" / "results"
    default_out_dir = repo_root / "memory_machines" / "training" / "optimized_tiering" / "viz"

    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(default_results_dir),
        help="Directory containing comparison JSONL files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(default_out_dir),
        help="Directory to save visualizations",
    )

    args = parser.parse_args()
    main(results_dir=args.results_dir, out_dir=args.out_dir)
