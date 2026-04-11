"""
Compare DSPy optimized program vs existing tiering classifier performance.

Evaluates both classifiers on the same highlights from the srs-highlights dataset
and outputs results in JSONL format for analysis.

Example usage:
    python -m memory_machines.training.optimized_tiering.compare_classifiers \
        --program-path memory_machines/training/checkpoints/optimized_tiering/20251222_133944/optimized_program.json \
        --model claude-sonnet-4-5 \
        --dspy-model anthropic/claude-sonnet-4-5 \
        --auth-token $ANTHROPIC_API_KEY
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import cast

import dspy
import pandas as pd
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI, Omit
from tqdm.asyncio import tqdm

from memory_machines.tiering.classify import classify_tier, format_reference_cards, format_source_context
from memory_machines.training.optimized_tiering.train import ClassifyMemoryPrompt
from memory_machines.utils.judge import SamplingParams
from memory_machines.utils.tiers import get_tier_from_task_type
from memory_machines.utils.types import Highlight


class _ClassificationResult(Highlight):
    """Result structure for classifier comparison experiment."""

    run_id: str
    target_content: str
    completion: str
    prediction_tier: str
    expected_tier: str


async def main(
    dataset: Dataset,
    program_path: str,
    dspy_model: str,
    dspy_api_key: str,
    tiering_client: AsyncOpenAI,
    tiering_params: SamplingParams,
    output_dir: Path,
    existing_results: Path,
):
    """Run comparison between DSPy program and tiering classifier."""

    # Configure DSPy LM
    logging.info(f"Configuring DSPy with model: {dspy_model}")
    lm = dspy.LM(dspy_model, temperature=0.0, max_tokens=4000, api_key=dspy_api_key)
    dspy.configure(lm=lm)

    # Load DSPy program
    logging.info(f"Loading DSPy program from: {program_path}")
    program = dspy.ChainOfThought(ClassifyMemoryPrompt)
    program.load(program_path)

    # Load existing results (incremental processing)
    if existing_results.exists():
        df_existing = pd.read_json(existing_results, lines=True)
        logging.info(f"Loaded {len(df_existing)} existing results")
    else:
        keys = list(_ClassificationResult.__annotations__.keys())
        df_existing = pd.DataFrame(columns=keys)  # type: ignore[reportArgumentType]
        logging.info("No existing results found, starting fresh")

    # Get already-processed task IDs
    existing_task_ids = df_existing["task_id"].unique() if not df_existing.empty else []

    # Filter out tasks that already have results
    dataset = dataset.filter(
        lambda x: x["task_id"] not in existing_task_ids,
        desc="Filtering out tasks that already have results",
    )
    logging.info(f"After filtering: {len(dataset)} highlights remaining")

    semaphore = asyncio.Semaphore(16)

    async def _process(row: Highlight) -> list[_ClassificationResult]:
        """Process a single highlight with both classifiers."""
        try:
            # Isolate the test target
            references = list(row["references"])
            target_index = next(i for i, ref in enumerate(references) if ref["test_target"])
            target = references.pop(target_index)

            # Format inputs for DSPy (same as prepare_dataset.py)
            source_description = format_source_context(row)
            reference_cards = format_reference_cards(references)
            target_prompt = target["content"]
            correct_tier = get_tier_from_task_type(target["task_type"])

            # DSPy classification (sync → async via executor)
            loop = asyncio.get_event_loop()
            dspy_prediction = await loop.run_in_executor(
                None,
                lambda: program(
                    source_description=source_description,
                    reference_cards=reference_cards,
                    target_prompt=target_prompt,
                ),
            )

            # Parse DSPy tier
            dspy_tier = dspy_prediction.tier.strip().upper()
            if dspy_tier not in ["T0", "T1", "T2", "T3"]:
                logging.error(f"Invalid DSPy tier '{dspy_tier}' for task {row['task_id']}")
                return []

            async with semaphore:
                # Tiering classification (async)
                tiering_tier, tiering_completion = await classify_tier(
                    client=tiering_client,
                    highlight=row,
                    references=references,
                    target=target["content"],
                    params=tiering_params,
                )

            # Validate tiering tier
            if tiering_tier not in ["T0", "T1", "T2", "T3"]:
                logging.error(f"Invalid tiering tier '{tiering_tier}' for task {row['task_id']}")
                return []

            # Build two result rows
            return [
                {
                    "run_id": "dspy-grounded",
                    **row,
                    "target_content": target["content"],
                    "completion": str(dspy_prediction),
                    "prediction_tier": dspy_tier,
                    "expected_tier": correct_tier,
                },
                {
                    "run_id": f"{tiering_params.model}-grounded",
                    **row,
                    "target_content": target["content"],
                    "completion": tiering_completion,
                    "prediction_tier": tiering_tier,
                    "expected_tier": correct_tier,
                },
            ]

        except Exception as e:
            logging.error(f"Error processing highlight {row.get('highlight_id', 'unknown')}: {e}. Skipping.")
            return []

    # Process all highlights
    classified_tasks_by_annotation = await tqdm.gather(
        *[_process(row=cast(Highlight, row)) for row in dataset],
        desc="Comparing classifiers",
    )
    classified_tasks = [t for r in classified_tasks_by_annotation for t in r]

    # Merge with existing results
    df = pd.concat([df_existing, pd.DataFrame(classified_tasks)], ignore_index=True)

    # Save to JSONL
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_json(existing_results, orient="records", lines=True)

    logging.info(f"Saved results to {existing_results}")
    logging.info(f"Total results: {len(df)} ({len(classified_tasks)} new)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Compare DSPy program vs tiering classifier performance")
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
        "--program-path",
        required=True,
        help="Path to DSPy program JSON file",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name for tiering classifier",
    )
    parser.add_argument(
        "--dspy-model",
        required=True,
        help="Model name for DSPy program",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for tiering classifier. If omitted, the SDK omits the field.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="Reasoning effort for tiering classifier.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom base URL for tiering classifier API",
    )
    parser.add_argument(
        "--auth-token",
        type=str,
        default=None,
        help="Custom authentication token for tiering classifier",
    )
    parser.add_argument(
        "--extra-body",
        type=str,
        default=None,
        help='Extra body parameters as JSON string (e.g., \'{"thinking": {"type": "enabled", "budget_tokens": 8194}}\')',
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="memory_machines/training/results/classifier_comparison",
        help="Output directory for results",
    )

    args = parser.parse_args()

    load_dotenv()

    # Load dataset
    logging.info(f"Loading dataset {args.dataset}, split={args.split}")
    dataset = load_dataset(args.dataset, split=args.split)
    assert isinstance(dataset, Dataset), "Dataset is not a Dataset"
    logging.info(f"Loaded {len(dataset)} highlights")

    # Use Omit if temperature not specified
    temperature = args.temperature if args.temperature is not None else Omit()

    # Parse extra-body JSON if provided
    extra_kwargs = {}
    if args.extra_body is not None:
        extra_kwargs["extra_body"] = json.loads(args.extra_body)
        logging.info(f"Parsed extra_body: {extra_kwargs}")

    # Setup tiering classifier sampling params
    sampling_params = SamplingParams(
        model=args.model,
        temperature=temperature,
        reasoning_effort=args.reasoning_effort if args.reasoning_effort is not None else Omit(),
        extra_kwargs=extra_kwargs,
    )
    logging.info(f"Tiering classifier: model={args.model}, temperature={temperature}")

    # Create tiering classifier client
    client_kwargs = {}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    else:
        # Default to Anthropic API
        client_kwargs["base_url"] = "https://api.anthropic.com/v1/"

    if args.auth_token:
        client_kwargs["api_key"] = args.auth_token

    tiering_client = AsyncOpenAI(**client_kwargs)

    # Get DSPy API key (uses same key as tiering client for Anthropic models)
    dspy_api_key = args.auth_token or os.getenv("ANTHROPIC_API_KEY")
    assert dspy_api_key is not None, (
        "DSPy API key must be provided via --auth-token or ANTHROPIC_API_KEY env var"
    )

    logging.info(f"DSPy: model={args.dspy_model}")

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_results = output_dir / f"comparison_{args.model}.jsonl"

    # Run comparison
    asyncio.run(
        main(
            dataset=dataset,
            program_path=args.program_path,
            dspy_model=args.dspy_model,
            dspy_api_key=dspy_api_key,
            tiering_client=tiering_client,
            tiering_params=sampling_params,
            output_dir=output_dir,
            existing_results=existing_results,
        )
    )
