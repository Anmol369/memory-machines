"""
Pluckability evaluation: Binary classification testing whether judges can distinguish pluckable
memory prompts (T2/T3: reviewable and well-suited to the highlight) from unpluckable ones
(T0/T1: off-target or requiring major refactoring).
"""

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, cast
from openai import AsyncOpenAI, Omit
from datasets import DatasetDict, load_dataset
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from memory_machines.utils.judge import SamplingParams, call_openai_judge
from memory_machines.utils.types import MemoryPrompt


async def evaluate_pluckability(
    client: AsyncOpenAI,
    instruction: str,
    params: SamplingParams,
    row: dict[str, Any],
):
    """
    Evaluate pluckability for a single row (one memory prompt).
    """
    typed_row = cast(MemoryPrompt, row)

    try:
        judge = await call_openai_judge(client=client, instruction=instruction, row=typed_row, params=params)
    except Exception as e:
        logging.warning(f"Error calling OpenAI API for id={typed_row.get('id', 'unknown')}: {e}")
        judge = None

    if judge is None:
        correct = None
    else:
        correct = judge["prediction"] == typed_row["pluckable"]

    return {
        "judge_prediction": judge["prediction"] if judge is not None else None,
        "judge_reason": judge["reason"] if judge is not None else None,
        "judge_usage": judge["usage"] if judge is not None else None,
        "correct": correct,
    }


async def evaluate_dataset(
    client: AsyncOpenAI,
    df: pd.DataFrame,
    instruction: str,
    params: SamplingParams,
    output_path: Path,
    max_concurrency: int,
    batch_size: int,
) -> None:
    """
    Evaluate all rows with missing `judge_prediction` in batches; after each batch, save `df` to disk.
    """

    semaphore = asyncio.Semaphore(max_concurrency)
    df = df.set_index("id", drop=False)

    pending_ids = df.loc[df["judge_prediction"].isna(), "id"].astype(int).tolist()
    logging.info(f"Found {len(pending_ids)} rows to evaluate (judge_prediction is None/NA).")

    async def process_one(row_id: int) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            row_dict = df.loc[row_id].to_dict()
            update = await evaluate_pluckability(
                client=client,
                instruction=instruction,
                params=params,
                row=row_dict,
            )
            return row_id, update

    with tqdm(total=len(pending_ids), desc="Evaluating") as pbar:
        for start in range(0, len(pending_ids), batch_size):
            batch_ids = pending_ids[start : start + batch_size]
            batch_results = await asyncio.gather(*(process_one(row_id) for row_id in batch_ids))

            for row_id, update in batch_results:
                for k, v in update.items():
                    df.at[row_id, k] = v

            # Save checkpoint after each batch
            df.to_json(output_path, orient="records", lines=True)
            pbar.update(len(batch_ids))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description=("Run an LLM judge over a dataset and write per-row pluckability predictions as JSONL.")
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
        "--instruction-file",
        type=str,
        required=True,
        help=(
            "Path to a system prompt file. The judge must output a JSON object containing exactly "
            "one boolean value (the verdict), and may include a 'reason' string."
        ),
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Maximum number of concurrent judge API calls.",
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

    instruction_name = args.instruction_file.split("/")[-1].split(".")[0]
    assert instruction_name.strip() != "", "Instruction name is empty"
    logging.info(f"Using instruction name: {instruction_name}")

    with open(args.instruction_file, "r") as f:
        system_instruction = f.read()

    assert system_instruction.strip() != "", "System instruction file is empty"

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

    # Build output filename (dataframe checkpoint)
    # Build output filename
    temp_suffix = f"_{args.temperature}" if args.temperature is not None else ""
    # Make model name filesystem-safe for filenames (common model IDs include "/" or ":" etc.)
    safe_model_name = (
        args.model.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_").replace("\t", "_")
    )
    # Dataframe checkpoint stored as JSON Lines (one row per record).
    output_file = f"results_{instruction_name}_{safe_model_name}{temp_suffix}.jsonl"

    # check if results directory exists
    repo_root = Path(__file__).resolve().parents[2]
    results_dir = repo_root / "memory_machines" / "pluckability" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / output_file

    # Load existing dataframe checkpoint or create an empty one with typed columns.
    base_columns = list(MemoryPrompt.__annotations__.keys())
    judge_columns = ["judge_prediction", "judge_reason", "judge_usage", "correct"]
    all_columns = ["split", *base_columns, *judge_columns]

    if output_path.exists():
        logging.info(f"Loading checkpoint dataframe from {output_path}")
        df = cast(pd.DataFrame, pd.read_json(output_path, orient="records", lines=True))
    else:
        df = pd.DataFrame(columns=pd.Index(all_columns))

    # If the dataframe is empty, initialize it from the dataset and mark all rows as unjudged.
    if df.empty:
        logging.info("No existing checkpoint found; initializing dataframe from dataset.")
        split_dfs: list[pd.DataFrame] = []
        for split_name in dataset.keys():
            split_name = str(split_name)
            split_df = cast(pd.DataFrame, dataset[split_name].to_pandas())
            split_df = split_df.copy()
            split_df["split"] = split_name
            split_dfs.append(split_df)

        df = cast(pd.DataFrame, pd.concat(split_dfs, ignore_index=True))
        for col in judge_columns:
            df[col] = None

        # Ensure column order / presence is stable
        for col in all_columns:
            if col not in df.columns:
                df[col] = None
        df = df[all_columns]

        df.to_json(output_path, orient="records", lines=True)
        logging.info(f"Wrote initialized dataframe checkpoint to {output_path} ({len(df)} rows).")

    df = cast(pd.DataFrame, df)
    asyncio.run(
        evaluate_dataset(
            client=client,
            df=df,
            instruction=system_instruction,
            params=sampling_params,
            output_path=output_path,
            max_concurrency=args.max_concurrency,
            batch_size=args.max_concurrency * 2,
        )
    )
