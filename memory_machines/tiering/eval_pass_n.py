"""
Pass@N evaluation: Evaluates pass@N accuracy for tier classification by sampling N
classifications per task. A task "passes" if at least one of the N samples predicts
the correct tier. This metric helps assess whether reinforcement learning approaches
like GRPO are viable by quantifying the "zero-gradient problem" (tasks where all N
samples are incorrect, leaving no positive signal to reinforce).
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import cast

import pandas as pd
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI, Omit
from tqdm.asyncio import tqdm

from memory_machines.tiering.classify import classify_tier
from memory_machines.utils.judge import SamplingParams
from memory_machines.utils.types import Highlight


def _sanitize_model_name(model: str) -> str:
    return model.replace("/", "_").replace("\\", "_")


def load_existing_results(existing_results: Path, n_samples: int) -> tuple[pd.DataFrame, set[str]]:
    """
    Load existing results and determine which tasks are complete.

    A task is considered complete if it has exactly n_samples results.
    Incomplete tasks are discarded and will be reprocessed.

    Args:
        existing_results: Path to existing results JSONL file
        n_samples: Expected number of samples per task

    Returns:
        (dataframe of complete results, set of completed task IDs)
    """
    if not existing_results.exists():
        return pd.DataFrame(), set()

    df = pd.read_json(existing_results, lines=True)
    assert isinstance(df, pd.DataFrame)

    # Filter out summary records if they exist
    if "summary" in df.columns:
        df = df[df["summary"] != True]  # noqa: E712

    if df.empty:
        return pd.DataFrame(), set()

    # Count samples per task
    task_sample_counts = df.groupby("task_id").size()

    # Tasks with exactly N samples are complete
    completed_tasks = set(task_sample_counts[task_sample_counts == n_samples].index)  # pyright: ignore[reportAttributeAccessIssue]

    # Keep only complete tasks
    df_complete = df[df["task_id"].isin(completed_tasks)]
    assert isinstance(df_complete, pd.DataFrame)

    logging.info(f"Loaded {len(completed_tasks)} completed tasks from existing results")
    if len(task_sample_counts) > len(completed_tasks):
        incomplete = len(task_sample_counts) - len(completed_tasks)
        logging.warning(f"Discarding {incomplete} incomplete tasks (will be reprocessed)")

    return df_complete, completed_tasks


def calculate_pass_at_n_metrics(df: pd.DataFrame, n_samples: int) -> dict:
    if df.empty:
        return {
            "total_tasks": 0,
            "pass_at_n_accuracy": 0.0,
            "zero_gradient_count": 0,
            "zero_gradient_pct": 0.0,
            "mean_correct_per_task": 0.0,
        }

    # Group by task_id
    task_groups = df.groupby("task_id")

    task_passed = []
    correct_per_task = []

    for task_id, task_df in task_groups:
        assert len(task_df) == n_samples
        predictions = task_df["prediction_tier"].tolist()
        expected = task_df["expected_tier"].iloc[0]  # Same for all samples

        correct_samples = [pred == expected for pred in predictions]
        n_correct = sum(correct_samples)

        task_passed.append(n_correct > 0)  # At least one correct
        correct_per_task.append(n_correct)

    # Calculate metrics
    n_tasks = len(task_passed)
    n_passed = sum(task_passed)
    n_zero_gradient = sum((c == n_samples) or (c == 0) for c in correct_per_task)

    return {
        "total_tasks": n_tasks,
        "pass_at_n_accuracy": n_passed / n_tasks if n_tasks > 0 else 0.0,
        "zero_gradient_count": n_zero_gradient,
        "zero_gradient_pct": n_zero_gradient / n_tasks if n_tasks > 0 else 0.0,
        "mean_correct_per_task": sum(correct_per_task) / n_tasks if n_tasks > 0 else 0.0,
    }


async def _process_task(
    row: Highlight,
    n_samples: int,
    client: AsyncOpenAI,
    params: SamplingParams,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    try:
        # Extract test target
        references = list(row["references"])
        target_index = next(i for i, ref in enumerate(references) if ref["test_target"])
        target = references.pop(target_index)

        # Sample N classifications concurrently within semaphore
        async with semaphore:
            samples = await asyncio.gather(
                *[
                    classify_tier(
                        client=client,
                        highlight=row,
                        references=references,  # Grounded with reference cards
                        target=target["content"],
                        params=params,
                    )
                    for _ in range(n_samples)
                ]
            )

        # Format results
        results = []
        for sample_idx, (pred_tier, completion) in enumerate(samples):
            results.append(
                {
                    "task_id": row["task_id"],
                    "highlight_id": row["highlight_id"],
                    "sample_idx": sample_idx,
                    "run_id": f"{params.model}-grounded-temp{params.temperature}-n{n_samples}",
                    "split": row["split"],
                    "source_url": row["source_url"],
                    "source_meta": row["source_meta"],
                    "highlight": row["highlight"],
                    "highlight_interpretation": row["highlight_interpretation"],
                    "references": references,
                    "target_content": target["content"],
                    "expected_tier": target["tier"],
                    "completion": completion,
                    "prediction_tier": pred_tier,
                }
            )

        return results

    except Exception as e:
        task_id = row.get("task_id", "unknown")
        logging.error(f"Error processing task {task_id}: {type(e).__name__}: {e}")
        logging.debug(f"Task {task_id} will be retried on next run with same parameters")
        # Return empty results - task will be retried on next run
        return []


async def main(
    dataset: Dataset,
    client: AsyncOpenAI,
    out_dir: Path,
    existing_results: Path,
    params: SamplingParams,
    n_samples: int = 8,
):
    """
    Run pass@N evaluation for tier classification.

    For each task in the dataset:
    1. Sample N tier classifications at the specified temperature
    2. Task "passes" if ANY sample predicts the correct tier
    3. Track zero-gradient tasks (all N samples incorrect)

    Args:
        dataset: Hugging Face dataset (grounded split)
        client: AsyncOpenAI client
        out_dir: Output directory for results
        existing_results: Path to existing results file for resume
        params: Sampling parameters
        n_samples: Number of samples per task (default: 8)
    """
    # Load existing results and determine completed tasks
    df_existing, completed_task_ids = load_existing_results(existing_results, n_samples)

    # Filter dataset to pending tasks
    dataset = dataset.filter(
        lambda x: x["task_id"] not in completed_task_ids,
        desc="Filtering completed tasks",
    )
    logging.info(
        f"Processing {len(dataset)} tasks ({n_samples} samples each = {len(dataset) * n_samples} total API calls)"
    )

    if len(dataset) == 0:
        logging.info("All tasks already completed!")
        # Still calculate and display metrics
        if not df_existing.empty:
            metrics = calculate_pass_at_n_metrics(df_existing, n_samples)
            logging.info(f"\n=== Pass@{n_samples} Results ===")
            logging.info(f"Total tasks: {metrics['total_tasks']}")
            logging.info(f"Pass@{n_samples} accuracy: {metrics['pass_at_n_accuracy']:.2%}")
            logging.info(
                f"Zero-gradient tasks: {metrics['zero_gradient_count']} ({metrics['zero_gradient_pct']:.2%})"
            )
            logging.info(f"Mean correct samples per task: {metrics['mean_correct_per_task']:.2f}")
        return

    # Process all tasks with incremental saving
    model_name_safe = _sanitize_model_name(params.model)
    output_file = out_dir / f"pass_n_results_{model_name_safe}_temp{params.temperature}_n{n_samples}.jsonl"
    semaphore = asyncio.Semaphore(2)  # Control API concurrency

    # Process in batches for incremental saving
    # Batch size set to allow good concurrency while checkpointing progress
    batch_size = 50  # Save progress every 50 tasks (~400 API calls with n=8)
    all_new_results = []

    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch = dataset.select(range(batch_start, batch_end))

        logging.info(
            f"Processing batch {batch_start // batch_size + 1} ({batch_start + 1}-{batch_end} of {len(dataset)} tasks)"
        )

        # Process batch
        batch_results = await tqdm.gather(
            *[_process_task(cast(Highlight, row), n_samples, client, params, semaphore) for row in batch],
            desc=f"Batch {batch_start // batch_size + 1}",
        )

        # Flatten batch results and track failures
        for i, task_results in enumerate(batch_results):
            if len(task_results) == 0:
                # Task failed - track it for retry
                task_id = batch[i]["task_id"]
                logging.warning(f"Task {task_id} failed - will be retried on next run")
            else:
                all_new_results.extend(task_results)

        # Incremental save after each batch
        if all_new_results:
            df_new = pd.DataFrame(all_new_results)
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
            df_all.to_json(output_file, orient="records", lines=True)
            logging.info(f"Saved {len(all_new_results)} results so far to {output_file}")

    # Final summary
    logging.info("\n=== Processing Summary ===")
    logging.info(f"Successfully processed: {len(all_new_results) // n_samples} tasks")

    # Load final results for metrics calculation
    df_all, _ = load_existing_results(output_file, n_samples)

    # Calculate and log metrics
    metrics = calculate_pass_at_n_metrics(df_all, n_samples)
    logging.info(f"\n=== Pass@{n_samples} Results ===")
    logging.info(f"Total tasks: {metrics['total_tasks']}")
    logging.info(f"Pass@{n_samples} accuracy: {metrics['pass_at_n_accuracy']:.2%}")
    logging.info(f"Zero-gradient tasks: {metrics['zero_gradient_count']} ({metrics['zero_gradient_pct']:.2%})")
    logging.info(f"Mean correct samples per task: {metrics['mean_correct_per_task']:.2f}")

    # Append summary to JSONL for programmatic access
    with open(output_file, "a") as f:
        summary = {
            "summary": True,
            "run_id": f"{params.model}-grounded-temp{params.temperature}-n{n_samples}",
            **metrics,
            "config": {
                "model": params.model,
                "temperature": float(params.temperature) if not isinstance(params.temperature, Omit) else None,
                "n_samples": n_samples,
                "reasoning_effort": params.reasoning_effort
                if not isinstance(params.reasoning_effort, Omit)
                else None,
            },
        }
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate pass@N accuracy for tier classification.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="laddermedia/srs-highlights",
        help="Hugging Face dataset name/path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="grounded",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5",
        help="Model name for classification",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for diversity (default: 1.0)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=8,
        help="Number of samples per task (default: 8)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="Reasoning effort for the model.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom base URL for OpenAI API",
    )
    parser.add_argument(
        "--auth-token",
        type=str,
        default=None,
        help="Custom authentication token",
    )
    parser.add_argument(
        "--extra-body",
        type=str,
        default=None,
        help='Extra body parameters as JSON string (e.g., \'{"thinking": {"type": "enabled", "budget_tokens": 8194}}\')',
    )
    args = parser.parse_args()

    load_dotenv()

    # Parse temperature
    temperature = args.temperature if args.temperature is not None else Omit()

    # Parse extra-body JSON if provided
    extra_kwargs = {}
    if args.extra_body is not None:
        try:
            extra_kwargs["extra_body"] = json.loads(args.extra_body)
            logging.info(f"Parsed extra_body: {extra_kwargs}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse --extra-body as JSON: {e}")

    sampling_params = SamplingParams(
        model=args.model,
        temperature=temperature,
        reasoning_effort=args.reasoning_effort if args.reasoning_effort is not None else Omit(),
        extra_kwargs=extra_kwargs,
    )
    logging.info(f"Sampling with model={args.model}, temperature={temperature}, n_samples={args.n_samples}")

    # Load dataset
    logging.info(f"Loading dataset {args.dataset}, split={args.split}")
    dataset = load_dataset(args.dataset, split=args.split)
    assert isinstance(dataset, Dataset), "Dataset is not a Dataset"

    # Create client
    client_kwargs = {}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    else:
        # Default to Anthropic API for this experiment
        client_kwargs["base_url"] = "https://api.anthropic.com/v1/"

    if args.auth_token:
        client_kwargs["api_key"] = args.auth_token
    else:
        # use the default API key for the model
        pass
    client = AsyncOpenAI(**client_kwargs)

    # Set up output directory
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "memory_machines" / "tiering" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name_safe = _sanitize_model_name(args.model)
    existing_results = (
        out_dir / f"pass_n_results_{model_name_safe}_temp{args.temperature}_n{args.n_samples}.jsonl"
    )

    # Run experiment
    asyncio.run(
        main(
            dataset=dataset,
            client=client,
            out_dir=out_dir,
            existing_results=existing_results,
            params=sampling_params,
            n_samples=args.n_samples,
        )
    )
