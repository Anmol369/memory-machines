#!/usr/bin/env python3
"""
Analyze probes evaluation results from JSONL files.

This script scans the memory_machines/probes/results folder for JSONL files and
displays performance metrics grouped by probe_name (probe type).

Usage:
    python -m memory_machines.probes.analyze                  # Display results as tables
    python -m memory_machines.probes.analyze --csv output.csv # Export results to CSV
"""

import argparse
from pathlib import Path
import warnings
from typing import Dict

import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def load_jsonl_data(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all JSONL files from results directory into DataFrames.

    Args:
        results_dir: Path to the results directory

    Returns:
        Dictionary mapping run names (filenames without .jsonl) to DataFrames
    """
    data = {}
    for jsonl_file in results_dir.glob("*.jsonl"):
        run_name = jsonl_file.stem  # filename without extension
        try:
            df = pd.read_json(jsonl_file, orient="records", lines=True)
            data[run_name] = df
        except Exception as e:
            print(f"Warning: Failed to load {jsonl_file}: {e}")
    return data


def calculate_metrics(df: pd.DataFrame) -> tuple[float, float, float, float, float, int, int, int, int]:
    """Calculate accuracy, precision, recall, F1 score, balanced accuracy, and confusion matrix.

    For probes:
    - Ground truth: probe_label == "positive" means True (should detect probe)
    - Prediction: judge_prediction

    Returns: (accuracy, precision, recall, f1, balanced_accuracy, tp, fp, tn, fn)
    """
    # Filter out rows where judge_prediction is None
    df_filtered = df[~pd.isna(df["judge_prediction"])].copy()
    assert isinstance(df_filtered, pd.DataFrame)
    assert len(df_filtered) > 0, "No rows found in filtered dataframe"

    # Convert labels to binary: "positive" = 1 (should detect), "negative" = 0 (should not detect)
    y_true = (df_filtered["probe_label"] == "positive").astype(int)
    y_pred = df_filtered["judge_prediction"].astype(int)

    try:
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division="warn")
        recall = recall_score(y_true, y_pred, zero_division="warn")
        f1 = f1_score(y_true, y_pred, zero_division="warn")

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except Exception:
        accuracy = precision = recall = f1 = balanced_acc = 0.0
        tp = fp = tn = fn = 0

    return accuracy, precision, recall, f1, balanced_acc, int(tp), int(fp), int(tn), int(fn)


def print_tables_by_probe(probes_data: Dict[str, list], console: Console) -> None:
    """Print performance tables grouped by probe_name.

    Args:
        probes_data: Dictionary mapping probe names to lists of run data
        console: Rich console for display
    """
    for probe_name in sorted(probes_data.keys()):
        runs_list = probes_data[probe_name]
        # Sort by F1 score (descending)
        runs_list.sort(key=lambda x: x["f1"], reverse=True)

        table = Table(
            title=f"Probe: {probe_name}",
            show_header=True,
            header_style="bold magenta",
            box=None,
            padding=(0, 1),
            title_justify="left",
        )

        table.add_column("Run", style="cyan", no_wrap=True)
        table.add_column("n", justify="right", style="green", no_wrap=True)
        table.add_column("pos/neg", justify="right", style="dim", no_wrap=True)
        table.add_column("accuracy", justify="right", style="yellow", no_wrap=True)
        table.add_column("bal_acc", justify="right", style="bold yellow", no_wrap=True)
        table.add_column("precision", justify="right", style="yellow", no_wrap=True)
        table.add_column("recall", justify="right", style="yellow", no_wrap=True)
        table.add_column("f1", justify="right", style="yellow", no_wrap=True)
        table.add_column("TP", justify="right", style="green", no_wrap=True)
        table.add_column("FP", justify="right", style="red", no_wrap=True)
        table.add_column("TN", justify="right", style="green", no_wrap=True)
        table.add_column("FN", justify="right", style="red", no_wrap=True)

        for run_data in runs_list:
            table.add_row(
                run_data["run"],
                str(run_data["n_total"]),
                f"{run_data['n_positive']}/{run_data['n_negative']}",
                f"{run_data['accuracy']:.3f}",
                f"{run_data['balanced_accuracy']:.3f}",
                f"{run_data['precision']:.3f}",
                f"{run_data['recall']:.3f}",
                f"{run_data['f1']:.3f}",
                str(run_data["tp"]),
                str(run_data["fp"]),
                str(run_data["tn"]),
                str(run_data["fn"]),
            )

        console.print(table)
        console.print()  # Add spacing between tables


def analyze_by_probe(data: Dict[str, pd.DataFrame], console: Console, csv_path: str | None = None) -> None:
    """Analyze performance by probe_name and either display as tables or export to CSV.

    Args:
        data: Dictionary mapping run names to DataFrames
        console: Rich console for logging
        csv_path: If provided, export to CSV instead of displaying tables
    """
    if not csv_path:
        console.print("\n[bold blue]🔍 Performance by Probe[/bold blue]")

    # Collect all probe names and their data
    probes_data = {}
    rows = []

    for run_name, df in data.items():
        if "probe_name" not in df.columns:
            console.print(f"[yellow]Warning: 'probe_name' column not found in {run_name}, skipping[/yellow]")
            continue

        probe_names = df["probe_name"].unique()
        for probe_name in probe_names:
            if probe_name not in probes_data:
                probes_data[probe_name] = []

            probe_df = df[df["probe_name"] == probe_name].copy()
            # Filter out None predictions
            probe_df_filtered = probe_df[~pd.isna(probe_df["judge_prediction"])]
            assert isinstance(probe_df_filtered, pd.DataFrame)

            if len(probe_df_filtered) > 0:
                total = len(probe_df_filtered)
                accuracy, precision, recall, f1, balanced_acc, tp, fp, tn, fn = calculate_metrics(
                    probe_df_filtered
                )

                # Count positive and negative examples
                n_positive = len(probe_df_filtered[probe_df_filtered["probe_label"] == "positive"])
                n_negative = len(probe_df_filtered[probe_df_filtered["probe_label"] == "negative"])

                run_data = {
                    "probe_name": probe_name,
                    "run": run_name,
                    "n_total": total,
                    "n_positive": n_positive,
                    "n_negative": n_negative,
                    "accuracy": accuracy,
                    "balanced_accuracy": balanced_acc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                }

                probes_data[probe_name].append(run_data)
                rows.append(run_data)

    if not probes_data:
        console.print("[yellow]No probe data found in any results[/yellow]")
        return

    # Export to CSV or display tables
    if csv_path:
        results_df = pd.DataFrame(rows)
        results_df = results_df.sort_values(["probe_name", "f1"], ascending=[True, False])
        results_df.to_csv(csv_path, index=False)
        console.print(f"[green]✅ Results exported to {csv_path}[/green]")
    else:
        print_tables_by_probe(probes_data, console)


def main():
    """Main analysis function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Analyze probes evaluation results from JSONL files")
    parser.add_argument(
        "--csv",
        type=str,
        help="Export results to CSV file instead of displaying tables",
        metavar="OUTPUT_FILE",
    )
    args = parser.parse_args()

    console = Console()

    # Get the results directory
    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "memory_machines" / "probes" / "results"

    if not results_dir.exists():
        console.print(f"[red]Error: Results directory not found at {results_dir}[/red]")
        return

    console.print(f"[blue]Loading JSONL files from {results_dir}[/blue]")

    # Load all JSONL data
    data = load_jsonl_data(results_dir)

    if not data:
        console.print("[red]No JSONL files found in results directory[/red]")
        return

    console.print(f"[green]Loaded {len(data)} JSONL files[/green]")

    # Analyze by probe_name (will either display tables or export to CSV)
    analyze_by_probe(data, console, csv_path=args.csv)

    if not args.csv:
        console.print("\n[bold green]✅ Analysis complete![/bold green]")


if __name__ == "__main__":
    main()
