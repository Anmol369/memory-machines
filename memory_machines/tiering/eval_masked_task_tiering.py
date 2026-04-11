"""
Masked task tiering: Evaluates whether providing reference cards (grounding) improves tier
classification accuracy. For each highlight, classifies the test target both ungrounded (without
reference cards) and grounded (with reference cards) to measure the impact of grounding context.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import cast

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from dotenv import load_dotenv
from openai import AsyncOpenAI, Omit
from tqdm.asyncio import tqdm

from memory_machines.tiering.classify import classify_tier
from memory_machines.utils.judge import SamplingParams
from memory_machines.utils.types import Highlight


class _ClassificationResult(Highlight):
    """Result structure for grounded/ungrounded comparison experiment."""

    run_id: str
    target_content: str
    completion: str
    prediction_tier: str
    expected_tier: str


def _log_tier_distribution(dataset: Dataset) -> None:
    """Log the distribution of expected tiers and well-posed tasks in the dataset."""
    tier_order = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}

    tier_counts: dict[str, int] = {}
    well_posed_counts: dict[str, int] = {}

    for row in dataset:
        assert isinstance(row, dict), "Row is not a dict"
        references = row["references"]

        # Find the test target and its tier
        expected_tier = None
        for ref in references:
            if ref.get("test_target"):
                expected_tier = ref.get("tier", "unknown")
                break

        if expected_tier is None:
            continue

        tier_counts[expected_tier] = tier_counts.get(expected_tier, 0) + 1

        # Check if well-posed: any non-target reference has tier >= expected tier
        reference_tiers = [ref.get("tier") for ref in references if not ref.get("test_target", False)]
        expected_rank = tier_order.get(expected_tier, -1)
        is_well_posed = any(tier_order.get(t, -1) >= expected_rank for t in reference_tiers)

        if is_well_posed:
            well_posed_counts[expected_tier] = well_posed_counts.get(expected_tier, 0) + 1

    # Sort tiers for consistent display
    sorted_tiers = sorted(tier_counts.keys())
    total = sum(tier_counts.values())
    total_well_posed = sum(well_posed_counts.values())

    logging.info("Task tier distribution:")
    logging.info("-" * 50)
    logging.info(f"{'Tier':<6} {'Count':>8} {'%':>8} {'Well-Posed':>12} {'%':>8}")
    logging.info("-" * 50)
    for tier in sorted_tiers:
        count = tier_counts[tier]
        pct = (count / total) * 100 if total > 0 else 0
        wp_count = well_posed_counts.get(tier, 0)
        wp_pct = (wp_count / count) * 100 if count > 0 else 0
        logging.info(f"{tier:<6} {count:>8} {pct:>7.1f}% {wp_count:>12} {wp_pct:>7.1f}%")
    logging.info("-" * 50)
    total_wp_pct = (total_well_posed / total) * 100 if total > 0 else 0
    logging.info(f"{'Total':<6} {total:>8} {'':>8} {total_well_posed:>12} {total_wp_pct:>7.1f}%")


async def main(
    dataset: Dataset,
    client: AsyncOpenAI,
    out_dir: Path,
    existing_results: Path,
    params: SamplingParams,
):
    """Run grounded vs ungrounded experiment."""
    # Load existing results if they exist
    if existing_results.exists():
        df_existing = pd.read_json(existing_results, lines=True)
    else:
        keys = list(_ClassificationResult.__annotations__.keys())
        df_existing = pd.DataFrame(columns=keys)  # type: ignore[reportArgumentType]

    semaphore = asyncio.Semaphore(16)

    if params.model.startswith("wandb-artifact://"):
        # special case for W&B artifacts: remove the artifact prefix
        model_name_safe = params.model.split("/")[-1].replace(":v1", "")
    else:
        model_name_safe = params.model

    async def _process(row: Highlight) -> list[_ClassificationResult]:
        """Process a single highlight with both grounded and ungrounded classification."""
        results: list[_ClassificationResult] = []

        try:
            # Isolate the test target
            references = list(row["references"])
            target_index = next(i for i, ref in enumerate(references) if ref["test_target"])
            target = references.pop(target_index)

            async with semaphore:
                # Classify without grounding
                ungrounded_tier, ungrounded_completion = await classify_tier(
                    client=client,
                    highlight=row,
                    references=[],
                    target=target["content"],
                    params=params,
                )
                results.append(
                    {
                        "run_id": f"{model_name_safe}-ungrounded",
                        **row,
                        "target_content": target["content"],
                        "completion": ungrounded_completion,
                        "prediction_tier": ungrounded_tier,
                        "expected_tier": target["tier"],
                    }
                )

                # Classify with grounding
                grounded_tier, grounded_completion = await classify_tier(
                    client=client,
                    highlight=row,
                    references=references,
                    target=target["content"],
                    params=params,
                )
                results.append(
                    {
                        "run_id": f"{model_name_safe}-grounded",
                        **row,
                        "target_content": target["content"],
                        "completion": grounded_completion,
                        "prediction_tier": grounded_tier,
                        "expected_tier": target["tier"],
                    }
                )
        except Exception as e:
            logging.error(f"Error processing highlight {row.get('highlight_id', 'unknown')}: {e}. Skipping.")
            # Return empty results to skip this highlight
            return []

        return results

    # Filter out tasks that already have results
    existing_task_ids = df_existing["task_id"].unique() if not df_existing.empty else []
    dataset = dataset.filter(
        lambda x: x["task_id"] not in existing_task_ids,
        desc="Filtering out tasks that already have results",
    )
    logging.info(f"After filtering: {len(dataset)} highlights remaining")

    # Process all highlights
    classified_tasks_by_annotation = await tqdm.gather(
        *[_process(row=cast(Highlight, row)) for row in dataset],
        desc="Classifying tasks",
    )
    classified_tasks = [t for r in classified_tasks_by_annotation for t in r]

    # Merge with existing results
    df = pd.concat([df_existing, pd.DataFrame(classified_tasks)], ignore_index=True)
    df.to_json(out_dir / f"classified_tasks_{model_name_safe}.jsonl", orient="records", lines=True)

    logging.info(f"Saved results to {out_dir / f'classified_tasks_{model_name_safe}.jsonl'}")
    logging.info(f"Total results: {len(df)} ({len(classified_tasks)} new)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Compare grounded vs ungrounded tier classification performance."
    )
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
        "--use-train-prompt",
        action="store_true",
        default=False,
        help="Use the training prompt for classification instead of the default prompt",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for the model. If omitted, the SDK omits the field.",
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
    parser.add_argument(
        "--heldout",
        type=str,
        default=None,
        help="Path to a local HuggingFace dataset (saved via save_to_disk) to concatenate with the main dataset.",
    )
    args = parser.parse_args()

    load_dotenv()

    # Use Omit if temperature not specified
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
    logging.info(f"Sampling with model: {args.model} and temperature: {temperature}")

    if args.use_train_prompt:
        logging.info("Using training prompt for classification")
        sampling_params.extra_kwargs["use_train_prompt"] = True

    # Load dataset
    logging.info(f"Loading dataset {args.dataset}, split={args.split}")
    dataset = load_dataset(args.dataset, split=args.split)
    assert isinstance(dataset, Dataset), "Dataset is not a Dataset"

    # Optionally concatenate with heldout dataset
    if args.heldout:
        logging.info(f"Loading heldout dataset from {args.heldout}")
        heldout_dataset_dict = load_from_disk(args.heldout)
        assert isinstance(heldout_dataset_dict, DatasetDict), "Heldout dataset is not a DatasetDict"
        assert "held_out" in heldout_dataset_dict.keys(), "Heldout dataset does not contain a held_out split"
        heldout_dataset = heldout_dataset_dict["held_out"]

        assert isinstance(heldout_dataset, Dataset), "Heldout dataset is not a Dataset"
        logging.info(
            f"Concatenating {len(dataset)} + {len(heldout_dataset)} = {len(dataset) + len(heldout_dataset)} highlights"
        )
        dataset = concatenate_datasets([dataset, heldout_dataset])

    # Log tier distribution
    _log_tier_distribution(dataset)

    # Create client
    client_kwargs = {}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url

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

    existing_results = out_dir / f"classified_tasks_{args.model}.jsonl"

    # Run experiment
    asyncio.run(
        main(
            dataset=dataset,
            client=client,
            out_dir=out_dir,
            existing_results=existing_results,
            params=sampling_params,
        )
    )
