#!/usr/bin/env python3
"""
Visualize tiering results framed as pluckability classification.

This script reads the masked tiering experiment results and frames them as binary pluckability
classification (T2/T3 = pluckable, T0/T1 = unpluckable) to evaluate grounded vs ungrounded
classification accuracy for the reviewability threshold.

Usage:
    python -m memory_machines.tiering.visualize_pluckability                  # Use default paths
    python -m memory_machines.tiering.visualize_pluckability --out-dir ./viz  # Custom output
"""

import argparse
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.metrics import accuracy_score, precision_score, recall_score


def tier_to_pluckable(tier: str) -> bool:
    """Convert tier label to pluckability boolean."""
    return tier in ["T2", "T3"]


def calculate_pluckability_metrics(df: pd.DataFrame) -> dict:
    """Calculate accuracy, precision, recall, and FPR for pluckability (reviewability)."""
    # Filter out rows with missing predictions
    df_filtered = df[~pd.isna(df["prediction_tier"])].copy()

    if len(df_filtered) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "fpr": 0.0,
            "n_total": 0,
        }

    # Convert tiers to pluckability
    y_true = df_filtered["expected_tier"].apply(tier_to_pluckable).astype(int)
    y_pred = df_filtered["prediction_tier"].apply(tier_to_pluckable).astype(int)

    try:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        # Calculate FPR: FP / (FP + TN) = FP / N (total actual negatives)
        # False Positives: predicted pluckable (1) but actually not pluckable (0)
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        # Total actual negatives (FP + TN)
        n_negatives = (y_true == 0).sum()
        fpr = fp / n_negatives if n_negatives > 0 else 0.0
    except Exception:
        accuracy = precision = recall = fpr = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "n_total": len(df_filtered),
    }


def calculate_pluckability_by_tier(df: pd.DataFrame) -> dict:
    """Calculate pluckable classification accuracy by expected tier."""
    # Filter out rows with missing predictions
    df_filtered = df[~pd.isna(df["prediction_tier"])].copy()

    if len(df_filtered) == 0:
        return {tier: {"accuracy": 0.0, "n": 0} for tier in ["T0", "T1", "T2", "T3"]}

    # Calculate pluckable classification accuracy for each expected tier
    tier_metrics = {}
    for tier in ["T0", "T1", "T2", "T3"]:
        tier_rows = df_filtered[df_filtered["expected_tier"] == tier]
        if len(tier_rows) > 0:
            # Expected pluckability for this tier
            expected_pluckable = tier_to_pluckable(tier)
            # Actual predictions converted to pluckability
            predicted_pluckable = tier_rows["prediction_tier"].apply(tier_to_pluckable)
            # Accuracy: how often did we correctly classify the pluckability?
            correct = (predicted_pluckable == expected_pluckable).sum()
            accuracy = correct / len(tier_rows)
            tier_metrics[tier] = {"accuracy": accuracy, "n": len(tier_rows)}
        else:
            tier_metrics[tier] = {"accuracy": 0.0, "n": 0}

    return tier_metrics


def visualize_precision_recall_scatter(metrics_by_run: dict, out_dir: str):
    """
    Create a precision-recall scatter plot comparing grounded vs ungrounded strategies.

    Args:
        metrics_by_run: Dictionary mapping run IDs to their pluckability metrics
        out_dir: Directory to save the visualization
    """
    # Extract data for plotting
    runs = []
    strategies = []
    precisions = []
    recalls = []
    accuracies = []

    for run_id, metrics in metrics_by_run.items():
        runs.append(run_id)
        # More robust strategy detection
        if run_id.endswith("-grounded"):
            strategy = "grounded"
        elif run_id.endswith("-ungrounded"):
            strategy = "ungrounded"
        else:
            raise ValueError(f"Unknown strategy: {run_id}")
        strategies.append(strategy)
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        accuracies.append(metrics["accuracy"])

    # Create figure with styling to match pluckability experiment
    fig, ax = plt.subplots(figsize=(10, 8))

    # Blue-tinted pastel colors (matching bar chart and pluckability experiment)
    strategy_colors = {"ungrounded": "#A8DADC", "grounded": "#457B9D"}

    # Create scatter plot by strategy
    for strategy in ["ungrounded", "grounded"]:
        strategy_mask = [s == strategy for s in strategies]
        strategy_recalls = [r for r, m in zip(recalls, strategy_mask) if m]
        strategy_precisions = [p for p, m in zip(precisions, strategy_mask) if m]

        if strategy_recalls:  # Only plot if there's data for this strategy
            ax.scatter(
                strategy_recalls,
                strategy_precisions,
                s=250,
                c=strategy_colors[strategy],
                alpha=0.9,
                edgecolors="white",
                linewidth=2,
                label=strategy.capitalize(),
                zorder=3,
            )

    # Add labels for each point
    for i, (run, strategy) in enumerate(zip(runs, strategies)):
        # Extract model name from run_id
        model_name = run.replace("-grounded", "").replace("-ungrounded", "")
        ax.annotate(
            model_name,
            (recalls[i], precisions[i]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10,
            color="#2C3E50",
            fontweight="500",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="#B0B0B0",
                alpha=0.95,
                linewidth=1,
            ),
            zorder=4,
        )

    # Customize the plot to match pluckability style
    ax.set_xlabel("Recall (Pluckable)", fontsize=13, fontweight="bold", color="#2C3E50")
    ax.set_ylabel("Precision (Pluckable)", fontsize=13, fontweight="bold", color="#2C3E50")
    ax.set_title(
        "Highlight-Conditioned Pluckability: Precision-Recall",
        fontsize=15,
        fontweight="bold",
        pad=20,
        color="#2C3E50",
    )

    # Set axis limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Grid styling to match pluckability
    ax.grid(True, alpha=0.2, linestyle="--", color="#B0B0B0", zorder=1)
    ax.set_facecolor("#FAFAFA")

    # Format axes as percentages
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.2, linewidth=1, zorder=2)

    # Add legend with custom styling
    legend_patches = [
        mpatches.Patch(color=strategy_colors["ungrounded"], label="Ungrounded", alpha=0.9),
        mpatches.Patch(color=strategy_colors["grounded"], label="Grounded", alpha=0.9),
    ]
    ax.legend(
        handles=legend_patches,
        fontsize=11,
        loc="lower right",
        framealpha=0.95,
    )

    plt.tight_layout()

    # Save figure
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "pluckability_precision_recall_scatter.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Precision-recall scatter plot saved to {output_path}")


def visualize_accuracy_bar_chart(metrics_by_run: dict, out_dir: str):
    """
    Create a bar chart comparing grounded vs ungrounded accuracy.

    Args:
        metrics_by_run: Dictionary mapping run IDs to their pluckability metrics
        out_dir: Directory to save the visualization
    """
    # Group by model and strategy
    models = set()
    for run_id in metrics_by_run.keys():
        model = run_id.replace("-grounded", "").replace("-ungrounded", "")
        models.add(model)

    models = sorted(models)

    # Prepare data
    grounded_acc = []
    ungrounded_acc = []

    for model in models:
        grounded_run = f"{model}-grounded"
        ungrounded_run = f"{model}-ungrounded"

        grounded_acc.append(metrics_by_run.get(grounded_run, {}).get("accuracy", 0))
        ungrounded_acc.append(metrics_by_run.get(ungrounded_run, {}).get("accuracy", 0))

    # Set up the bar chart
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))

    # Blue-tinted pastel color palette (matching pluckability experiment)
    color_ungrounded = "#A8DADC"  # Light blue pastel
    color_grounded = "#457B9D"  # Deeper blue pastel

    # Create bars with rounded corners
    def create_rounded_bars(x_positions, heights, width, color, label):
        bars = []
        for i, (x_pos, height) in enumerate(zip(x_positions, heights)):
            # Create a rounded rectangle patch
            rounded_rect = mpatches.FancyBboxPatch(
                (x_pos - width / 2, 0),
                width,
                height,
                boxstyle=mpatches.BoxStyle("Round", pad=0.02),
                edgecolor="white",
                facecolor=color,
                linewidth=1.5,
                alpha=0.9,
            )
            ax.add_patch(rounded_rect)
            bars.append(rounded_rect)
        return bars

    create_rounded_bars(x - width / 2, ungrounded_acc, width, color_ungrounded, "Ungrounded")
    create_rounded_bars(x + width / 2, grounded_acc, width, color_grounded, "Grounded")

    # Add value labels above bars
    def add_value_labels(x_positions, heights, offset):
        for x_pos, height in zip(x_positions, heights):
            if height > 0:  # Only show label if there's data
                ax.text(
                    x_pos + offset,
                    height + 0.02,  # Small padding above the bar
                    f"{height:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="#2C3E50",
                )

    add_value_labels(x, ungrounded_acc, -width / 2)
    add_value_labels(x, grounded_acc, width / 2)

    # Customize the plot
    ax.set_xlabel("Model", fontsize=13, fontweight="bold", color="#2C3E50")
    ax.set_ylabel("Accuracy", fontsize=13, fontweight="bold", color="#2C3E50")
    ax.set_title(
        "Highlight-Conditioned Pluckability: Ungrounded vs Grounded",
        fontsize=15,
        fontweight="bold",
        pad=20,
        color="#2C3E50",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)

    # Create custom legend with rounded patches
    legend_patches = [
        mpatches.Patch(color=color_ungrounded, label="Ungrounded", alpha=0.9),
        mpatches.Patch(color=color_grounded, label="Grounded", alpha=0.9),
    ]
    ax.legend(
        handles=legend_patches,
        fontsize=11,
        loc="upper right",
        framealpha=0.95,
    )

    # Set axis limits and formatting
    ax.set_xlim(-0.5, len(models) - 0.5)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", alpha=0.2, linestyle="--", color="#B0B0B0")
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()

    # Save figure
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, "pluckability_accuracy_bar_chart.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Accuracy bar chart saved to {output_path}")


def display_pluckability_by_tier_table(pluckability_by_tier_by_run: dict):
    """
    Display a Rich table showing pluckable classification accuracy by expected tier.

    Args:
        pluckability_by_tier_by_run: Dictionary mapping run IDs to their tier-specific metrics
    """
    console = Console()

    # Create table
    table = Table(
        title="Pluckable Classification Accuracy by Expected Tier", show_header=True, header_style="bold"
    )
    table.add_column("Run ID", style="cyan")
    table.add_column("Strategy", style="magenta")
    table.add_column("T0 Acc", justify="right")
    table.add_column("T0 n", justify="right", style="dim")
    table.add_column("T1 Acc", justify="right")
    table.add_column("T1 n", justify="right", style="dim")
    table.add_column("T2 Acc", justify="right")
    table.add_column("T2 n", justify="right", style="dim")
    table.add_column("T3 Acc", justify="right")
    table.add_column("T3 n", justify="right", style="dim")

    # Add rows for each run
    for run_id, tier_metrics in pluckability_by_tier_by_run.items():
        # Strategy detection
        if run_id.endswith("-grounded"):
            strategy = "Grounded"
        elif run_id.endswith("-ungrounded"):
            strategy = "Ungrounded"
        else:
            strategy = "Grounded" if "grounded" in run_id.lower() else "Ungrounded"

        table.add_row(
            run_id,
            strategy,
            f"{tier_metrics['T0']['accuracy']:.1%}",
            str(tier_metrics["T0"]["n"]),
            f"{tier_metrics['T1']['accuracy']:.1%}",
            str(tier_metrics["T1"]["n"]),
            f"{tier_metrics['T2']['accuracy']:.1%}",
            str(tier_metrics["T2"]["n"]),
            f"{tier_metrics['T3']['accuracy']:.1%}",
            str(tier_metrics["T3"]["n"]),
        )

    console.print(table)


def main(results_dir: str, out_dir: str):
    """Scan results directory and generate pluckability visualizations from tiering results."""
    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    # Find all classified_tasks_*.jsonl files
    result_files = [
        f for f in os.listdir(results_dir) if f.startswith("classified_tasks") and f.endswith(".jsonl")
    ]

    if not result_files:
        print(f"No classified_tasks files found in {results_dir}")
        return

    print(f"Found {len(result_files)} result file(s) in {results_dir}:")
    for f in result_files:
        print(f"  - {f}")

    # Calculate pluckability metrics for each run
    metrics_by_run = {}
    pluckability_by_tier_by_run = {}

    for result_file in result_files:
        results_path = os.path.join(results_dir, result_file)
        try:
            df = pd.read_json(results_path, lines=True)
            print(f"\nLoaded {len(df)} results from {result_file}")

            # Process by run_id (grounded/ungrounded)
            if "run_id" in df.columns:
                run_ids = df["run_id"].unique()
                for run_id in run_ids:
                    run_df = df[df["run_id"] == run_id].copy()

                    # Calculate pluckability metrics
                    metrics = calculate_pluckability_metrics(run_df)
                    metrics_by_run[run_id] = metrics

                    # Calculate pluckability accuracy by expected tier
                    tier_metrics = calculate_pluckability_by_tier(run_df)
                    pluckability_by_tier_by_run[run_id] = tier_metrics

                    strategy = "grounded" if "grounded" in run_id else "ungrounded"
                    print(f"  [{run_id}] {strategy.capitalize()}:")
                    print(f"    Accuracy: {metrics['accuracy']:.2%}")
                    print(f"    Precision: {metrics['precision']:.2%}")
                    print(f"    Recall: {metrics['recall']:.2%}")
                    print(f"    FPR: {metrics['fpr']:.2%}")
                    print(f"    n={metrics['n_total']}")

        except Exception as e:
            print(f"Error loading {result_file}: {e}")

    if not metrics_by_run:
        print("No valid result files could be loaded")
        return

    print(f"\n{'=' * 60}")
    print(f"Generating pluckability visualizations for {len(metrics_by_run)} runs")
    print(f"{'=' * 60}")

    # Display pluckability by tier table
    print()
    display_pluckability_by_tier_table(pluckability_by_tier_by_run)

    # Generate visualizations
    print()
    visualize_precision_recall_scatter(metrics_by_run, out_dir)
    visualize_accuracy_bar_chart(metrics_by_run, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize tiering results framed as pluckability classification"
    )

    # Get default paths relative to this script
    repo_root = Path(__file__).resolve().parents[2]
    default_results_dir = repo_root / "memory_machines" / "tiering" / "results"
    default_out_dir = repo_root / "memory_machines" / "tiering" / "viz"

    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(default_results_dir),
        help="Directory containing tiering results JSONL files",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(default_out_dir),
        help="Directory to save visualizations",
    )

    args = parser.parse_args()
    main(results_dir=args.results_dir, out_dir=args.out_dir)
