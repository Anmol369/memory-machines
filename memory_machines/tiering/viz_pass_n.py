"""
Visualize pass@N evaluation results in a formatted table.

Reads pass@N result files from the results directory and displays metrics
in a table matching the format from the essay (Table: pass@8 accuracy for GRPO viability).
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table
from memory_machines.tiering.eval_pass_n import calculate_pass_at_n_metrics


def _load_and_calculate_metrics(results_file: Path, n_samples: int) -> dict[str, Any] | None:
    """
    Load results data and calculate metrics, ignoring any stored summary records.

    This ensures metrics are always fresh and correctly calculated from raw data.

    Args:
        results_file: Path to the pass_n_results_*.jsonl file
        n_samples: Number of samples per task

    Returns:
        Dictionary with calculated metrics and config, or None if no data
    """
    if not results_file.exists():
        return None

    try:
        df = pd.read_json(results_file, lines=True)
        assert isinstance(df, pd.DataFrame)

        # Filter out summary records - we'll recalculate from raw data
        if "summary" in df.columns:
            df = df[df["summary"] != True]  # noqa: E712

        if df.empty or "task_id" not in df.columns:
            return None

        # Calculate metrics from raw data
        metrics = calculate_pass_at_n_metrics(df, n_samples)

        # Calculate pass@1 (accuracy using only the first sample)
        if "sample_idx" in df.columns:
            first_samples = df[df["sample_idx"] == 0]
            if not first_samples.empty:
                correct = first_samples["prediction_tier"] == first_samples["expected_tier"]
                metrics["pass_at_1_accuracy"] = correct.sum() / len(first_samples)
            else:
                metrics["pass_at_1_accuracy"] = 0.0
        else:
            metrics["pass_at_1_accuracy"] = 0.0

        # Extract model name from filename
        # Format: pass_n_results_{model}_temp{temp}_n{n}.jsonl
        filename_parts = results_file.stem.replace("pass_n_results_", "").split("_temp")
        model_name = filename_parts[0] if filename_parts else "unknown"

        # Add config info
        metrics["config"] = {"model": model_name}

        return metrics

    except Exception as e:
        return None


def format_model_name(model: str) -> str:
    """
    Format model name for display in table.

    Maps technical model names to human-readable names matching the essay format.
    """
    name_mapping = {
        "llama3.1-8b": "Llama 3.1 8B",
        "llama-3.1-8b": "Llama 3.1 8B",
        "meta-llama-3.1-8b": "Llama 3.1 8B",
        "gpt-oss-120b": "GPT-OSS 120B",
        "qwen-3-32b": "Qwen3 32B",
        "qwen3-32b": "Qwen3 32B",
        "deepseek-ai/deepseek-v3.1": "DeepSeek V3.1",
        "deepseek-v3.1": "DeepSeek V3.1",
        "claude-sonnet-4-5": "Claude Sonnet 4.5",
        "gpt-4o": "GPT-4o",
        "gpt-5.2": "GPT-5.2",
        "gemini-2.5-pro": "Gemini 2.5 Pro",
    }

    # Check for exact match
    if model in name_mapping:
        return name_mapping[model]

    # Check for partial match (case-insensitive)
    model_lower = model.lower()
    for key, value in name_mapping.items():
        if key in model_lower:
            return value

    # Return original if no mapping found
    return model


def create_pass_n_table(results_dir: Path, n_samples: int = 8) -> Table:
    """
    Create a Rich table displaying pass@N results.

    Args:
        results_dir: Directory containing pass_n_results_*.jsonl files
        n_samples: Number of samples per task (for column headers)

    Returns:
        Rich Table object ready to display
    """
    # Create table matching essay format
    table = Table(title=f"Pass@{n_samples} Accuracy for GRPO Viability Assessment")

    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Number of tasks", justify="right", style="magenta")
    table.add_column("pass@1 (%)", justify="right", style="blue")
    table.add_column(f"pass@{n_samples} (passed)", justify="right", style="green")
    table.add_column(f"pass@{n_samples} (%)", justify="right", style="yellow")
    table.add_column("Zero-gradient samples", justify="right", style="red")

    # Find all pass@N result files
    result_files = list(results_dir.glob(f"pass_n_results_*_n{n_samples}.jsonl"))

    if not result_files:
        # Return empty table with placeholder row
        table.add_row("No results found", "--", "--", "--", "--", "--")
        return table

    # Sort files by model name for consistent ordering
    result_files.sort()

    # Load and display results for each model
    for result_file in result_files:
        # Always calculate metrics from raw data
        metrics = _load_and_calculate_metrics(result_file, n_samples)

        if metrics is None:
            # File exists but no valid data (empty or corrupted)
            model_name = result_file.stem.replace("pass_n_results_", "").split("_temp")[0]
            table.add_row(
                format_model_name(model_name),
                "--",
                "--",
                "--",
                "--",
                "--",
                style="dim",
            )
            continue

        # Extract model name and metrics
        config = metrics.get("config", {})
        model = config.get("model", "unknown")
        total_tasks = metrics.get("total_tasks", 0)
        pass_at_1_accuracy = metrics.get("pass_at_1_accuracy", 0.0)
        pass_at_n_accuracy = metrics.get("pass_at_n_accuracy", 0.0)
        zero_gradient_count = metrics.get("zero_gradient_count", 0)

        # Calculate number of tasks that passed (at least one correct sample)
        tasks_passed = int(pass_at_n_accuracy * total_tasks)

        # Format model name
        display_name = format_model_name(model)

        # Add row to table
        table.add_row(
            display_name,
            str(total_tasks),
            f"{pass_at_1_accuracy * 100:.1f}%",
            str(tasks_passed),
            f"{pass_at_n_accuracy * 100:.1f}%",
            str(zero_gradient_count),
        )

    return table


def main():
    parser = argparse.ArgumentParser(description="Visualize pass@N evaluation results in a formatted table.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing pass_n_results_*.jsonl files (default: memory_machines/tiering/results)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=8,
        help="Number of samples per task to display (default: 8)",
    )
    args = parser.parse_args()

    # Determine results directory
    if args.results_dir is None:
        # Default to repo root / memory_machines / tiering / results
        script_dir = Path(__file__).resolve().parent
        results_dir = script_dir / "results"
    else:
        results_dir = args.results_dir

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Create and display table
    console = Console()
    table = create_pass_n_table(results_dir, n_samples=args.n_samples)
    console.print()
    console.print(table)
    console.print()

    # Print additional statistics if any results found
    result_files = list(results_dir.glob(f"pass_n_results_*_n{args.n_samples}.jsonl"))
    if result_files:
        console.print(f"📊 Loaded results from {len(result_files)} model(s)", style="bold green")
        console.print(f"📁 Results directory: {results_dir}", style="dim")
        console.print()
        console.print(
            "Note: Metrics are calculated from raw data (summary records are ignored)",
            style="cyan dim",
        )
    else:
        console.print(f"⚠️  No pass@{args.n_samples} results found in {results_dir}", style="bold yellow")
        console.print(
            "   Run the evaluation first: python -m memory_machines.tiering.eval_pass_n",
            style="dim",
        )


if __name__ == "__main__":
    main()
