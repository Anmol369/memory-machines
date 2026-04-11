#!/usr/bin/env python3
"""
Analyze forced-choice evaluation results from JSONL files.

This script scans the forced-choice results directory for JSONL files and
displays performance metrics including accuracy and task type distribution
of chosen cards.

By default, all the sources which are in the few-shot instruction are excluded from analysis.

Usage:
    python -m memory_machines.forced_choice.analyze                           # Excludes blocked sources (default)
    python -m memory_machines.forced_choice.analyze --include-blocked         # Includes blocked sources
"""

import argparse
import json
from pathlib import Path
from typing import Dict, TypedDict
import warnings

import pandas as pd
from datasets import Dataset, load_dataset
from rich.console import Console
from rich.table import Table
from sklearn.metrics import accuracy_score

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class ChooseFourMetrics(TypedDict):
    """Metrics calculated from forced-choice predictions."""

    accuracy: float
    n_correct: int
    n_total: int


def load_jsonl_data(
    results_dir: Path,
    dataset: Dataset | None = None,
    excluded_sources: list[str] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Load all JSONL files from results directory into DataFrames.

    Args:
        results_dir: Path to the results directory
        dataset: Optional dataset to map highlight_id to source_url
        excluded_sources: Optional list of source URLs to exclude from analysis

    Returns:
        Dictionary mapping run names (filenames without .jsonl) to DataFrames
    """
    data = {}

    # Create mapping from highlight_id to source_url if dataset provided
    highlight_to_source = {}
    if dataset is not None:
        for highlight in dataset:
            highlight_to_source[highlight["highlight_id"]] = highlight["source_url"]

    for jsonl_file in results_dir.glob("*.jsonl"):
        run_name = jsonl_file.stem  # filename without extension
        try:
            df = pd.read_json(jsonl_file, orient="records", lines=True)

            # Filter out excluded sources if provided
            if excluded_sources and dataset is not None:
                initial_count = len(df)
                # Add source_url column based on highlight_id
                df["source_url"] = df["highlight_id"].map(highlight_to_source)
                # Filter out excluded sources
                df = df[~df["source_url"].isin(excluded_sources)]
                # Drop the temporary source_url column
                df = df.drop(columns=["source_url"])
                filtered_count = initial_count - len(df)
                if filtered_count > 0:
                    print(f"Filtered {filtered_count} records from {run_name} (excluded sources)")

            data[run_name] = df
        except Exception as e:
            print(f"Warning: Failed to load {jsonl_file}: {e}")
    return data


def calculate_accuracy_metrics(df: pd.DataFrame) -> ChooseFourMetrics:
    """Calculate accuracy metrics for forced-choice task.

    Args:
        df: DataFrame with 'correct' column indicating whether the chosen card was pluckable

    Returns:
        Dictionary with accuracy metrics
    """
    # Filter out rows where correct is None
    df_filtered = df[~pd.isna(df["correct"])].copy()

    if len(df_filtered) == 0:
        return ChooseFourMetrics(
            accuracy=0.0,
            n_correct=0,
            n_total=0,
        )

    n_correct = df_filtered["correct"].sum()
    n_total = len(df_filtered)
    accuracy = accuracy_score([True] * n_total, df_filtered["correct"])

    return ChooseFourMetrics(
        accuracy=accuracy,
        n_correct=int(n_correct),
        n_total=int(n_total),
    )


def create_overall_performance_summary(data: Dict[str, pd.DataFrame], console: Console) -> None:
    """Create Overall Performance Summary table."""
    console.print("\n[bold blue]📊 Overall Performance Summary[/bold blue]")

    table = Table(
        title="Classification Accuracy (Choosing Pluckable Card)",
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 1),
        title_justify="left",
    )

    table.add_column("Run", style="cyan", no_wrap=True)
    table.add_column("n", justify="right", style="green", no_wrap=True)
    table.add_column("accuracy", justify="right", style="yellow", no_wrap=True)
    table.add_column("correct", justify="right", style="green", no_wrap=True)
    table.add_column("incorrect", justify="right", style="red", no_wrap=True)

    # Calculate metrics for all runs and sort by accuracy (descending)
    run_metrics = []
    for run_name, df in data.items():
        metrics = calculate_accuracy_metrics(df)
        run_metrics.append((run_name, metrics))

    # Sort by accuracy in descending order
    run_metrics.sort(key=lambda x: x[1]["accuracy"], reverse=True)

    # Add rows to table in sorted order
    for run_name, metrics in run_metrics:
        table.add_row(
            run_name,
            str(metrics["n_total"]),
            f"{metrics['accuracy']:.3f}",
            str(metrics["n_correct"]),
            str(metrics["n_total"] - metrics["n_correct"]),
        )

    console.print(table)


def create_task_type_distribution(data: Dict[str, pd.DataFrame], console: Console) -> None:
    """Create table showing distribution of task types for chosen cards."""
    console.print("\n[bold blue]📋 Task Type Distribution of Chosen Cards[/bold blue]")

    # Collect all task types and calculate percentages for each run
    all_task_types = set()
    run_distributions = {}

    for run_name, df in data.items():
        # Filter out None task types
        df_filtered = df[~pd.isna(df["chosen_task_type"])].copy()

        if len(df_filtered) == 0:
            console.print(f"[yellow]No task type data available for {run_name}[/yellow]")
            continue

        # Count task types
        task_type_series = df_filtered["chosen_task_type"]
        assert isinstance(task_type_series, pd.Series)
        task_type_counts = task_type_series.value_counts()
        total = len(df_filtered)

        # Store percentages for this run
        run_distributions[run_name] = {
            "counts": task_type_counts,
            "total": total,
            "percentages": {task_type: (count / total) * 100 for task_type, count in task_type_counts.items()},
        }

        # Track all task types
        all_task_types.update(task_type_counts.index)

    if not run_distributions:
        console.print("[yellow]No task type data available[/yellow]")
        return

    # Group runs by instruction type
    instruction_groups = {}
    for run_name in run_distributions.keys():
        # Extract instruction type from run name (e.g., "multi_shot", "zero_shot", "simple")
        instruction_type = (
            run_name.split("_")[0] + "_" + run_name.split("_")[1]
            if len(run_name.split("_")) > 1
            else run_name.split("_")[0]
        )
        # Handle edge cases where instruction is just one word
        if instruction_type.endswith("_shot"):
            instruction_type = instruction_type
        elif run_name.startswith("simple"):
            instruction_type = "simple"
        else:
            # Default to first part of the name
            instruction_type = run_name.split("_")[0]

        if instruction_type not in instruction_groups:
            instruction_groups[instruction_type] = []
        instruction_groups[instruction_type].append(run_name)

    # Sort task types alphabetically for consistent display
    sorted_task_types = sorted(all_task_types)

    # Create a separate table for each instruction type
    for instruction_type in sorted(instruction_groups.keys()):
        runs_in_group = instruction_groups[instruction_type]

        table = Table(
            title=f"Task Type Distribution - {instruction_type.upper()} (Percentages)",
            show_header=True,
            header_style="bold magenta",
            box=None,
            padding=(0, 1),
            title_justify="left",
        )

        # Add run column (first column)
        table.add_column("Run", style="cyan", no_wrap=True)

        # Add a column for each task type
        for task_type in sorted_task_types:
            table.add_column(str(task_type), justify="right", style="yellow", no_wrap=True)

        # Add rows for each run
        for run_name in runs_in_group:
            # Shorten the run name by removing the instruction prefix
            short_name = run_name.replace(instruction_type + "_", "", 1)
            row_data = [short_name]
            percentages = run_distributions[run_name]["percentages"]
            for task_type in sorted_task_types:
                if task_type in percentages:
                    row_data.append(f"{percentages[task_type]:.1f}%")
                else:
                    row_data.append("0.0%")
            table.add_row(*row_data)

        # Add total row
        total_row = ["[bold]TOTAL[/bold]"]
        for _ in sorted_task_types:
            total_row.append("[bold]100.0%[/bold]")
        table.add_row(*total_row, style="bold")

        console.print(table)
        console.print()


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description="Analyze forced-choice evaluation results from JSONL files.")
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
    parser.add_argument(
        "--dataset",
        type=str,
        default="laddermedia/srs-highlights",
        help="Dataset to load highlights from (default: laddermedia/srs-highlights)",
    )
    args = parser.parse_args()

    console = Console()

    # Get the results directory (default location from choose_most_preferred.py)
    script_dir = Path(__file__).resolve().parent
    results_dir = script_dir / "results"

    if not results_dir.exists():
        console.print(f"[red]Error: Results directory not found at {results_dir}[/red]")
        return

    # Load dataset for mapping highlight_id to source_url
    dataset = None
    excluded_sources = None

    if not args.include_blocked:
        console.print(f"[blue]Loading dataset from {args.dataset}[/blue]")
        try:
            dataset = load_dataset(args.dataset, split="grounded")
            assert isinstance(dataset, Dataset), "Dataset is not a Dataset"
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load dataset: {e}[/yellow]")
            console.print("[yellow]Proceeding without source filtering[/yellow]")
            dataset = None

        # Load blocked sources
        if dataset is not None:
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
    data = load_jsonl_data(results_dir, dataset, excluded_sources)

    if not data:
        console.print("[red]No JSONL files found in results directory[/red]")
        return

    console.print(f"[green]Loaded {len(data)} JSONL files[/green]")

    # Create summary tables
    create_overall_performance_summary(data, console)
    create_task_type_distribution(data, console)

    console.print("\n[bold green]✅ Analysis complete![/bold green]")


if __name__ == "__main__":
    main()
