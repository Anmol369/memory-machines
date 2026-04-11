"""
Probe evaluation: Binary classification testing whether judges can detect specific quality
issues in memory prompts. For each probe, judges evaluate positive examples (prompts tagged
with that issue) and negative examples (high-quality prompts) to measure detection accuracy.
"""

import argparse
import asyncio
from collections import defaultdict
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal, cast
from openai import AsyncOpenAI, Omit
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from memory_machines.utils.judge import SamplingParams, call_openai_judge
from memory_machines.utils.types import MemoryPrompt
from memory_machines.probes._probes import PROBES


def _prepare_negative_rows(dataset: DatasetDict) -> list[MemoryPrompt]:
    pluckable_rows = dataset.filter(lambda x: x["pluckable"], desc="Filtering for pluckable rows")
    assert isinstance(pluckable_rows, DatasetDict), "Pluckable rows is not a DatasetDict"

    # Sort split names for deterministic concatenation order
    pluckable_dataset = concatenate_datasets([pluckable_rows[split] for split in sorted(pluckable_rows.keys())])
    assert isinstance(pluckable_dataset, Dataset), "Pluckable dataset is not a Dataset"

    pluckable_dataset = pluckable_dataset.shuffle(seed=42)

    casted = cast(list[MemoryPrompt], pluckable_dataset.to_list())
    assert all([row["pluckable"] for row in casted]), "All rows are not pluckable"

    return casted


def _prepare_positive_rows(dataset: DatasetDict) -> dict[str, list[MemoryPrompt]]:
    # identify all the tagged rows in the dataset
    tagged_datasets = dataset.filter(lambda x: len(x["tags"]) > 0, desc="Filtering for tagged rows")
    assert isinstance(tagged_datasets, DatasetDict), "Tagged datasets is not a DatasetDict"

    # collect all the rows by tag (indiscriminately of their assigned split)
    # Sort split names for deterministic iteration order
    rows_by_tag: dict[str, list[MemoryPrompt]] = defaultdict(list)
    for split in sorted(tagged_datasets.keys()):
        for row in tagged_datasets[split]:
            assert isinstance(row, dict), type(row)
            tags = row.get("tags", [])
            assert isinstance(tags, list)
            for tag in tags:
                assert isinstance(tag, str)
                rows_by_tag[tag].append(cast(MemoryPrompt, row))

    return rows_by_tag


class _ProbableResult(MemoryPrompt):
    """
    positive: The card is a positive example of the probe.
    negative: The card is a negative example of the probe.
    """

    probe_label: Literal["positive", "negative"]
    probe_name: str

    judge_prediction: bool | None
    judge_reason: str | None


async def evaluate_probes(
    client: AsyncOpenAI,
    df: pd.DataFrame,
    dataset: DatasetDict,
    params: SamplingParams,
    output_path: Path,
    max_concurrency: int = 10,
    batch_size: int = 20,
):
    """Evaluate probes with checkpointing support."""

    # If dataframe is empty, initialize it with all test cases
    if df.empty:
        logging.info("Initializing test cases for all probes...")
        pos_rows_by_tag = _prepare_positive_rows(dataset)
        logging.info(f"Found {len(pos_rows_by_tag)} tags: {list(pos_rows_by_tag.keys())}")

        # Determine the maximum number of positive examples across all probes
        max_positive_count = 0
        for probe in PROBES:
            pos_cards = pos_rows_by_tag.get(probe.probe_name, [])
            if len(pos_cards) > max_positive_count:
                max_positive_count = len(pos_cards)

        logging.info(f"Max positive examples across all probes: {max_positive_count}")

        # Prepare a shared set of negative rows that all probes will use
        all_neg_rows = _prepare_negative_rows(dataset)
        shared_neg_rows: list[_ProbableResult] = [
            {
                **row,
                "probe_name": "<TBD>",  # this will be filled in the loop below
                "probe_label": "negative",
                "judge_prediction": None,
                "judge_reason": None,
            }
            for row in all_neg_rows[:max_positive_count]
        ]

        logging.info(f"Using {len(shared_neg_rows)} shared negative rows for all probes")

        all_test_cases = []

        # Create test cases for each probe
        for probe in PROBES:
            pos_cards = pos_rows_by_tag.get(probe.probe_name, [])
            if len(pos_cards) == 0:
                logging.warning(f"No rows found for {probe.probe_name}; skipping")
                continue

            pos_rows: list[_ProbableResult] = [
                {
                    **card,
                    "probe_name": probe.probe_name,
                    "probe_label": "positive",
                    "judge_prediction": None,
                    "judge_reason": None,
                }
                for card in pos_cards
            ]

            # Use the same shared negative rows for all probes
            neg_test_cases = [{**row, "probe_name": probe.probe_name} for row in shared_neg_rows]

            all_test_cases.extend(pos_rows + neg_test_cases)

        df = pd.DataFrame(all_test_cases)
        # Add a unique test_case_id for indexing
        df["test_case_id"] = range(len(df))
        df.to_json(output_path, orient="records", lines=True)
        logging.info(f"Initialized {len(df)} test cases and saved to {output_path}")

    df = df.set_index("test_case_id", drop=False)

    # Evaluate each probe
    for probe in PROBES:
        # Get rows for this probe that haven't been evaluated yet
        probe_mask = (df["probe_name"] == probe.probe_name) & (df["judge_prediction"].isna())
        pending_ids = df.loc[probe_mask, "test_case_id"].astype(int).tolist()

        if len(pending_ids) == 0:
            logging.info(f"No pending evaluations for {probe.probe_name}; skipping")
            continue

        logging.info(f"Evaluating {probe.probe_name}: {len(pending_ids)} test cases pending")

        # Use absolute path for instruction file (instruction_path is relative to repo root)
        repo_root = Path(__file__).resolve().parents[2]
        instruction_file = repo_root / probe.instruction_path
        with open(instruction_file, "r") as f:
            instruction = f.read()
        assert instruction.strip() != "", "Instruction is empty"

        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_one(test_case_id: int) -> tuple[int, dict[str, Any]]:
            async with semaphore:
                row_dict = df.loc[test_case_id].to_dict()
                card = cast(MemoryPrompt, row_dict)
                try:
                    judged_result = await call_openai_judge(
                        client=client,
                        instruction=instruction,
                        row=card,
                        params=params,
                    )
                except Exception as e:
                    logging.warning(f"Error calling OpenAI API for test_case_id {test_case_id}: {e}")
                    return test_case_id, {"judge_prediction": None, "judge_reason": None}

                if judged_result is None:
                    logging.warning(f"Got None prediction for test_case_id {test_case_id}")
                    return test_case_id, {"judge_prediction": None, "judge_reason": None}

                return test_case_id, {
                    "judge_prediction": judged_result["prediction"],
                    "judge_reason": judged_result["reason"],
                }

        # Process in batches
        with tqdm(total=len(pending_ids), desc=f"Evaluating {probe.probe_name}") as pbar:
            for start in range(0, len(pending_ids), batch_size):
                batch_ids = pending_ids[start : start + batch_size]
                batch_results = await asyncio.gather(*(process_one(test_case_id) for test_case_id in batch_ids))

                for test_case_id, update in batch_results:
                    for k, v in update.items():
                        df.at[test_case_id, k] = v

                # Save checkpoint after each batch
                df.to_json(output_path, orient="records", lines=True)
                pbar.update(len(batch_ids))

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Run LLM judge over probes and write per-test-case predictions as JSONL."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="laddermedia/srs-prompts",
        help="Hugging Face dataset name/path to load (must be a DatasetDict with splits).",
    )
    parser.add_argument("--model", type=str, required=True, help="Judge model name.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for the judge model. If omitted, the SDK omits the field.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Maximum number of concurrent judge API calls.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of test cases to evaluate before saving checkpoint. Defaults to max_concurrency * 2.",
    )
    parser.add_argument("--base-url", type=str, default=None, help="Custom base URL for OpenAI API")
    parser.add_argument("--auth-token", type=str, default=None, help="Custom authentication token")
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="Reasoning effort for the judge model.",
    )
    parser.add_argument(
        "--extra-body",
        type=str,
        default=None,
        help='Extra body parameters as a JSON string (e.g., \'{"thinking": {"type": "enabled", "budget_tokens": 2000}}\')',
    )
    args = parser.parse_args()

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

    load_dotenv()

    # Build client kwargs based on provided arguments
    client_kwargs = {}
    if args.base_url is not None:
        client_kwargs["base_url"] = args.base_url
    if args.auth_token is not None:
        client_kwargs["api_key"] = args.auth_token
    else:
        # Only assert API key is set if no auth token provided
        assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"

    client = AsyncOpenAI(**client_kwargs)

    logging.info(f"Loading dataset from {args.dataset}")
    dataset = load_dataset(args.dataset)
    assert isinstance(dataset, DatasetDict), "Dataset is not a DatasetDict"

    # Build output filename
    temp_suffix = f"_{args.temperature}" if args.temperature is not None else ""
    # Make model name filesystem-safe for filenames
    safe_model_name = (
        args.model.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_").replace("\t", "_")
    )
    output_file = f"results_probes_{safe_model_name}{temp_suffix}.jsonl"

    # Use absolute path for results directory
    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "memory_machines" / "probes" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / output_file

    # Load existing checkpoint or create empty DataFrame
    if output_path.exists():
        logging.info(f"Loading checkpoint from {output_path}")
        df = cast(pd.DataFrame, pd.read_json(output_path, orient="records", lines=True))
    else:
        df = pd.DataFrame()

    batch_size = args.batch_size if args.batch_size is not None else args.max_concurrency * 2

    df = asyncio.run(
        evaluate_probes(
            client=client,
            df=df,
            dataset=dataset,
            params=sampling_params,
            output_path=output_path,
            max_concurrency=args.max_concurrency,
            batch_size=batch_size,
        )
    )

    logging.info(f"Evaluation complete. Final results written to {output_path} ({len(df)} test cases)")
