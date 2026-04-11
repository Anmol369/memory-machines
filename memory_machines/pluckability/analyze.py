#!/usr/bin/env python3
"""
Analyze pluckability evaluation results from JSONL files.

This script scans the pluckability/results folder for JSONL files and
displays performance metrics including accuracy, precision, recall, and F1.

By default, all the sources which are in the few-shot instruction are excluded from analysis.

Usage:
    python3 analyze.py                           # Excludes blocked sources (default)
    python3 analyze.py --include-blocked         # Includes blocked sources
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, TypedDict

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


class MetricsResult(TypedDict):
    """Metrics calculated from predictions."""

    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float
    tp: int
    fp: int
    tn: int
    fn: int


def load_jsonl_data(results_dir: Path, excluded_sources: list[str] | None = None) -> Dict[str, pd.DataFrame]:
    """
    Load all JSONL files from the results directory.

    Args:
        results_dir: Path to the results directory
        excluded_sources: Optional list of source URLs to exclude from analysis

    Returns a dictionary mapping run names (derived from filename) to DataFrames.
    """
    data = {}

    for file in results_dir.glob("*.jsonl"):
        # Extract run name from filename (e.g., "results_zero_shot_gpt-5.2.jsonl" -> "zero_shot_gpt-5.2")
        run_name = file.stem.replace("results_", "")

        try:
            df = pd.read_json(file, orient="records", lines=True)

            # Filter out excluded sources if provided
            if excluded_sources:
                initial_count = len(df)
                df = df[~df["source_url"].isin(excluded_sources)]
                filtered_count = initial_count - len(df)
                if filtered_count > 0:
                    print(f"Filtered {filtered_count} records from {run_name} (excluded sources)")

            data[run_name] = df
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue

    return data


def calculate_metrics(df: pd.DataFrame) -> MetricsResult:
    """Calculate accuracy, balanced accuracy, precision, recall, F1 score, FPR, and confusion matrix."""
    # Filter out rows where judge_prediction is None
    df_filtered = df[~pd.isna(df["judge_prediction"])].copy()

    if len(df_filtered) == 0:
        return MetricsResult(
            accuracy=0.0,
            balanced_accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            fpr=0.0,
            tp=0,
            fp=0,
            tn=0,
            fn=0,
        )

    y_true = df_filtered["pluckable"].astype(int)
    y_pred = df_filtered["judge_prediction"].astype(int)

    try:
        accuracy = accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division="warn")
        recall = recall_score(y_true, y_pred, zero_division="warn")
        f1 = f1_score(y_true, y_pred, zero_division="warn")

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate FPR (False Positive Rate): FP / (FP + TN)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    except Exception:
        accuracy = balanced_accuracy = precision = recall = f1 = fpr = 0.0
        tp = fp = tn = fn = 0

    return MetricsResult(
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
    )


def create_overall_performance_summary(data: Dict[str, pd.DataFrame], console: Console) -> None:
    """Create Overall Performance Summary table."""
    console.print("\n[bold blue]📊 Overall Performance Summary[/bold blue]")

    table = Table(
        title="Overall Performance Summary",
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 1),
        title_justify="left",
    )

    table.add_column("Run", style="cyan", no_wrap=True)
    table.add_column("n", justify="right", style="green", no_wrap=True)
    table.add_column("accuracy", justify="right", style="yellow", no_wrap=True)
    table.add_column("balanced_acc", justify="right", style="yellow", no_wrap=True)
    table.add_column("precision", justify="right", style="yellow", no_wrap=True)
    table.add_column("recall", justify="right", style="yellow", no_wrap=True)
    table.add_column("f1", justify="right", style="yellow", no_wrap=True)
    table.add_column("fpr", justify="right", style="yellow", no_wrap=True)

    for run_name, df in sorted(data.items()):
        # Filter out None predictions for counting
        df_filtered = df[~pd.isna(df["judge_prediction"])]
        total_n = len(df_filtered)

        metrics = calculate_metrics(df)

        table.add_row(
            run_name,
            str(total_n),
            f"{metrics['accuracy']:.3f}",
            f"{metrics['balanced_accuracy']:.3f}",
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['f1']:.3f}",
            f"{metrics['fpr']:.3f}",
        )

    console.print(table)


def create_pluckability_breakdown(data: Dict[str, pd.DataFrame], console: Console) -> None:
    """Create breakdown by pluckability (pluckable vs unpluckable)."""
    console.print("\n[bold blue]📊 Pluckability Distribution[/bold blue]")

    table = Table(
        title="Pluckable vs Unpluckable Counts",
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 1),
        title_justify="left",
    )

    table.add_column("Run", style="cyan", no_wrap=True)
    table.add_column("Pluckable", justify="right", style="green", no_wrap=True)
    table.add_column("Unpluckable", justify="right", style="red", no_wrap=True)
    table.add_column("Total", justify="right", style="yellow", no_wrap=True)

    for run_name, df in sorted(data.items()):
        # Filter out None predictions
        df_filtered = df[~pd.isna(df["judge_prediction"])]

        if len(df_filtered) == 0:
            continue

        pluckable_count = (df_filtered["pluckable"] == True).sum()
        unpluckable_count = (df_filtered["pluckable"] == False).sum()
        total_count = len(df_filtered)

        table.add_row(
            run_name,
            str(pluckable_count),
            str(unpluckable_count),
            str(total_count),
        )

    console.print(table)
    console.print()  # Add spacing


def create_task_type_breakdown(data: Dict[str, pd.DataFrame], console: Console) -> None:
    """Create breakdown by task type across all splits."""
    console.print("\n[bold blue]📋 Task Type Performance Breakdown[/bold blue]")

    # Add task_type column to all dataframes
    data_with_task_type = {}
    for run_name, df in data.items():
        if "task_type" not in df.columns:
            console.print(
                f"[yellow]Warning: 'task_type' column not found in {run_name}, skipping task type analysis[/yellow]"
            )
            continue

        # drop rows with None task type
        df_none = df[df["task_type"].isna()]
        if len(df_none) > 0:
            console.print(f"[dim]Ignoring {len(df_none)} rows with None task type in {run_name}[/dim]")
        df_clean = df[df["task_type"].notna()].copy()
        data_with_task_type[run_name] = df_clean

    if not data_with_task_type:
        console.print("[yellow]No task type data available for analysis[/yellow]")
        return

    # Collect task type data across all splits
    task_type_data = {}

    for run_name, df in data_with_task_type.items():
        # Filter out None predictions
        df_filtered = df[~pd.isna(df["judge_prediction"])].copy()

        if len(df_filtered) == 0:
            continue

        # Calculate accuracy for each task type
        for task_type in df_filtered["task_type"].unique():
            task_df = df_filtered[df_filtered["task_type"] == task_type]

            if task_type not in task_type_data:
                task_type_data[task_type] = {"runs": {}, "n": len(task_df)}

            metrics = calculate_metrics(task_df)
            task_type_data[task_type]["runs"][run_name] = metrics["accuracy"]

    if not task_type_data:
        console.print("[yellow]No task type data available[/yellow]")
        return

    # Create unified table (inverted: runs as rows, task types as columns)
    table = Table(
        title="Accuracy by Task Type (All Splits)",
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 1),
        title_justify="left",
    )

    table.add_column("Run", style="cyan", no_wrap=True)

    # Add a column for each task type with row count
    task_types = sorted(task_type_data.keys())
    for task_type in task_types:
        n = task_type_data[task_type]["n"]
        table.add_column(f"{task_type}\n(n={n})", justify="right", style="yellow", no_wrap=True)

    # Add rows for each run
    run_names = sorted(data_with_task_type.keys())
    for run_name in run_names:
        row_data = [run_name]

        # Add accuracy for each task type
        for task_type in task_types:
            accuracy = task_type_data[task_type]["runs"].get(run_name)
            if accuracy is not None:
                row_data.append(f"{accuracy:.3f}")
            else:
                row_data.append("-")

        table.add_row(*row_data)

    console.print(table)
    console.print()  # Add spacing


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze pluckability evaluation results from JSONL files.")
    parser.add_argument(
        "--include-blocked",
        action="store_true",
        help="Include blocked/contaminated sources in analysis (by default they are excluded)",
    )
    parser.add_argument(
        "--blocked-sources-file",
        type=str,
        default="./instructions/few_shot_sources.json",
        help="Path to JSON file containing list of blocked source URLs (default: ./instructions/few_shot_sources.json)",
    )
    args = parser.parse_args()

    console = Console()

    # Get the results directory
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "results"

    if not results_dir.exists():
        console.print(f"[red]Error: Results directory not found at {results_dir}[/red]")
        return

    # Load blocked sources unless --include-blocked is set
    excluded_sources = None
    if not args.include_blocked:
        exclude_path = Path(args.blocked_sources_file)
        if not exclude_path.is_absolute():
            exclude_path = script_dir / exclude_path

        if exclude_path.exists():
            with open(exclude_path, "r") as f:
                exclude_data = json.load(f)
                excluded_sources = exclude_data.get("contaminated_sources", [])
            console.print(
                f"[yellow]Excluding {len(excluded_sources)} blocked/contaminated sources (use --include-blocked to include them)[/yellow]"
            )
        else:
            console.print(
                f"[yellow]Warning: Blocked sources file not found at {exclude_path}, proceeding without exclusions[/yellow]"
            )
    else:
        console.print("[yellow]Including blocked/contaminated sources in analysis[/yellow]")

    console.print(f"[blue]Loading JSONL files from {results_dir}[/blue]")

    # Load all JSONL data
    data = load_jsonl_data(results_dir, excluded_sources)

    if not data:
        console.print("[red]No JSONL files found in results directory[/red]")
        return

    console.print(f"[green]Loaded {len(data)} JSONL files[/green]")

    # Create summary tables
    create_overall_performance_summary(data, console)
    create_pluckability_breakdown(data, console)
    create_task_type_breakdown(data, console)

    console.print("\n[bold green]✅ Analysis complete![/bold green]")


if __name__ == "__main__":
    main()
