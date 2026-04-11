#!/usr/bin/env python3
"""
Visualize confusion matrices for tiering classification results.

Scans the memory_machines/tiering/results folder for JSONL files and generates
confusion matrix visualizations comparing grounded vs ungrounded tier classification performance.

Usage:
    python -m memory_machines.tiering.viz_confusion_matrices                    # Use default paths
    python -m memory_machines.tiering.viz_confusion_matrices --out-dir ./custom # Custom output
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def visualize_confusion_matrices(df: pd.DataFrame, out_dir: str):
    """
    Visualize confusion matrices for tiering results.

    Creates one graph per model run with two subplots (grounded and ungrounded strategies).

    Args:
        df: DataFrame containing classification results with columns:
            - run_id: The run identifier (e.g., "claude-sonnet-4-5-grounded")
            - prediction_tier: The predicted tier (T0, T1, T2, T3)
            - expected_tier: The actual tier (T0, T1, T2, T3)
        out_dir: Directory to save the visualizations
    """
    # Extract model and strategy from run_id
    df["model"] = df["run_id"].str.rsplit("-", n=1).str[0]
    df["strategy"] = df["run_id"].str.rsplit("-", n=1).str[-1]

    # Define the order of tiers for consistent confusion matrices
    tier_order = ["T0", "T1", "T2", "T3"]

    # Group by model
    models = df["model"].unique()

    os.makedirs(out_dir, exist_ok=True)

    for model in models:
        model_df = df[df["model"] == model]
        safe_model_name = model.replace("/", "_").replace(":", "_")

        # Create figure with two subplots (one for each strategy)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for idx, strategy in enumerate(["ungrounded", "grounded"]):
            strategy_df = model_df[model_df["strategy"] == strategy]
            subplot_json_path = os.path.join(out_dir, f"confusion_matrix_{safe_model_name}_{strategy}.json")

            if len(strategy_df) == 0:
                # No data for this strategy, skip it
                axes[idx].text(
                    0.5, 0.5, f"No data for {strategy}", ha="center", va="center", transform=axes[idx].transAxes
                )
                axes[idx].set_title(f"{strategy.capitalize()}", fontsize=12, fontweight="bold")

                # Persist an empty payload so every subplot has a JSON artifact.
                empty_payload = {
                    "model": model,
                    "strategy": strategy,
                    "labels": tier_order,
                    "matrix": [[0 for _ in tier_order] for _ in tier_order],
                    "total": 0,
                    "correct": 0,
                    "accuracy": 0.0,
                    "sample_count": 0,
                    "has_data": False,
                }
                with open(subplot_json_path, "w", encoding="utf-8") as f:
                    json.dump(empty_payload, f, indent=2)
                continue

            # Get predictions and expected values
            y_true = strategy_df["expected_tier"].to_numpy()  # type: ignore
            y_pred = strategy_df["prediction_tier"].to_numpy()  # type: ignore

            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=tier_order)

            # Calculate accuracy
            correct = int((y_true == y_pred).sum())  # type: ignore
            total = len(y_true)
            accuracy = correct / total if total > 0 else 0

            # Plot heatmap
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=tier_order,
                yticklabels=tier_order,
                ax=axes[idx],
                cbar=True,
                vmin=0,
            )

            # Set labels and title
            axes[idx].set_xlabel("Predicted", fontsize=10)
            axes[idx].set_ylabel("Expected", fontsize=10)
            axes[idx].set_title(
                f"{strategy.capitalize()}\nAccuracy: {accuracy:.2%} ({correct}/{total})",
                fontsize=12,
                fontweight="bold",
            )

            # Rotate labels for better readability
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha="right", fontsize=9)
            axes[idx].set_yticklabels(axes[idx].get_yticklabels(), rotation=0, fontsize=9)

            subplot_payload = {
                "model": model,
                "strategy": strategy,
                "labels": tier_order,
                "matrix": cm.tolist(),
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "sample_count": total,
                "has_data": True,
            }
            with open(subplot_json_path, "w", encoding="utf-8") as f:
                json.dump(subplot_payload, f, indent=2)

        # Add overall title
        fig.suptitle(f"Confusion Matrices: {model}", fontsize=14, fontweight="bold", y=1.02)

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(out_dir, f"confusion_matrix_{safe_model_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Confusion matrix visualization saved to {output_path}")


def calculate_per_tier_accuracy(df: pd.DataFrame, out_dir: str):
    """
    Calculate per-tier accuracy for each run and visualize as a heatmap table.

    Args:
        df: DataFrame containing classification results
        out_dir: Directory to save the visualization
    """
    # Define tier order
    tier_order = ["T0", "T1", "T2", "T3"]

    # Group by run_id
    runs = df["run_id"].unique()

    # Calculate accuracy for each tier in each run
    accuracy_data = []
    for run_id in runs:
        run_df = df[df["run_id"] == run_id]
        row_data = {"run_id": run_id}

        # Calculate Cohen's kappa for this run
        y_true = run_df["expected_tier"].to_numpy()
        y_pred = run_df["prediction_tier"].to_numpy()
        row_data["kappa"] = cohen_kappa_score(y_true, y_pred, weights="linear")

        for tier in tier_order:
            # Filter for rows where expected_tier is this tier
            tier_df = run_df[run_df["expected_tier"] == tier]
            if len(tier_df) > 0:
                # Calculate accuracy: % of times this tier was correctly predicted
                correct = (tier_df["prediction_tier"] == tier).sum()
                total = len(tier_df)
                accuracy = (correct / total) * 100
                row_data[tier] = accuracy
            else:
                row_data[tier] = None  # No samples for this tier

        accuracy_data.append(row_data)

    # Create DataFrame
    accuracy_df = pd.DataFrame(accuracy_data)
    accuracy_df = accuracy_df.set_index("run_id")

    # Reorder columns: kappa first, then tiers
    accuracy_df = accuracy_df[["kappa"] + tier_order]

    # Sort by run_id for better organization
    accuracy_df = accuracy_df.sort_index()

    # Save as CSV
    csv_path = os.path.join(out_dir, "per_tier_accuracy.csv")
    accuracy_df.to_csv(csv_path)
    print(f"\nPer-tier accuracy table saved to {csv_path}")

    # Create visualization (exclude kappa from heatmap - different scale)
    fig, ax = plt.subplots(figsize=(10, max(6, len(runs) * 0.5)))

    # Create heatmap
    sns.heatmap(
        accuracy_df[tier_order],
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        ax=ax,
        cbar_kws={"label": "Accuracy (%)"},
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor="gray",
    )

    # Set labels and title
    ax.set_xlabel("Tier", fontsize=12, fontweight="bold")
    ax.set_ylabel("Run ID", fontsize=12, fontweight="bold")
    ax.set_title("Per-Tier Classification Accuracy (%)", fontsize=14, fontweight="bold", pad=20)

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(out_dir, "per_tier_accuracy.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Per-tier accuracy visualization saved to {output_path}")

    # Print the table to console
    print("\nPer-Tier Accuracy Table (kappa = Cohen's kappa, linear weighted):")
    print(accuracy_df.to_string(float_format=lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"))


def _filter_well_posed_tasks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out poorly posed tasks where no reference has a tier >= expected tier.

    A task is "poorly posed" if the model is asked to predict a tier (e.g., T1) but all
    reference cards have lower tiers (e.g., only T0), making it impossible for the model
    to learn from the grounding context what the expected tier or higher looks like.
    """
    tier_order = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}

    def is_well_posed(row) -> bool:
        expected_tier = row["expected_tier"]
        assert expected_tier in tier_order
        references = row["references"]
        reference_tiers = [ref.get("tier") for ref in references if not ref.get("test_target", False)]

        expected_rank = tier_order.get(expected_tier, -1)
        # Well-posed if any reference tier >= expected tier
        return any(tier_order.get(t, -1) >= expected_rank for t in reference_tiers)

    mask = df.apply(is_well_posed, axis=1)
    filtered_df = df[mask].copy()
    assert isinstance(filtered_df, pd.DataFrame)

    return filtered_df


def main(results_dir: str, out_dir: str, include_poorly_posed: bool = False):
    """Scan results directory and generate confusion matrix visualizations for all result files."""
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    # Find all .jsonl files in the results directory
    result_files = [
        f for f in os.listdir(results_dir) if f.endswith(".jsonl") and f.startswith("classified_tasks")
    ]

    if not result_files:
        print(f"No result files found in {results_dir}")
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
            # df = df[df["split"] != "train"]
            print(f"\nLoaded {len(df)} results from {result_file}")

            # Show breakdown by model and strategy
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

    # Filter out poorly posed tasks unless --include-poorly-posed is set
    if not include_poorly_posed:
        filtered_df = _filter_well_posed_tasks(combined_df)
        excluded_count = len(combined_df) - len(filtered_df)
        combined_df = filtered_df
        print(
            f"Filtered to well-posed tasks: {len(combined_df)} results ({excluded_count} poorly-posed excluded)"
        )
    else:
        print("Including all tasks (including poorly-posed)")

    print(f"{'=' * 60}")

    # Generate visualizations
    visualize_confusion_matrices(combined_df, out_dir)

    # Generate per-tier accuracy table
    calculate_per_tier_accuracy(combined_df, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize confusion matrices for tiering results")

    # Get default paths relative to this script
    repo_root = Path(__file__).resolve().parents[2]
    default_results_dir = repo_root / "memory_machines" / "tiering" / "results"
    default_out_dir = repo_root / "memory_machines" / "tiering" / "viz"

    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(default_results_dir),
        help="Directory containing results JSONL files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(default_out_dir),
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--include-poorly-posed",
        action="store_true",
        help="Include poorly-posed tasks where the expected tier is not represented in reference cards",
    )

    args = parser.parse_args()
    main(results_dir=args.results_dir, out_dir=args.out_dir, include_poorly_posed=args.include_poorly_posed)
