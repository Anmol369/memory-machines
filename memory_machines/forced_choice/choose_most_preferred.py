"""
Forced-choice evaluation: Judges select the best memory prompt from a set of 4 candidates
(1 high-quality T3 prompt + 3 lower-quality T1/T2 prompts). This tests whether judges can
reliably identify high-quality memory prompts when presented with mixed-quality options.
"""

import argparse
import asyncio
import json
from typing import Any, TypedDict, cast
import pandas as pd
import logging
import os
import random
import re
from datasets import Dataset, load_dataset
from openai import AsyncOpenAI, Omit
from tqdm import tqdm
from memory_machines.utils.format import format_highlight_text
from memory_machines.utils.judge import SamplingParams
from memory_machines.utils.types import Highlight, HighlightReference

_JUDGE_CARD_ID_REGEX = r"Chosen: \*\*card_id=([a-z0-9]+)\*\*"

_CHOOSE_FOUR_PROMPT_TEMPLATE = """I am reading [{title}]({url}) by {author}

**Highlight:**

I highlighted the following text:
> {highlight}


**Interpretation of the highlight within the context of the document:**
{highlight_interpretation}

Which of these cards should I add to my Anki deck?

{cards}
"""

_alphabet = "abcdefghijklmnopqrstuvwxyz"


def _randomly_identify_cards(
    cards: list[HighlightReference], rng: random.Random
) -> dict[str, HighlightReference]:
    def _generate_id() -> str:
        random_letter = rng.choice(_alphabet)
        random_number = rng.randint(0, 9)
        return f"{random_letter}{random_number}"

    card_by_id: dict[str, HighlightReference] = {}
    for card in cards:
        id = _generate_id()
        while id in card_by_id:
            id = _generate_id()
        card_by_id[id] = card
    return card_by_id


async def choose_most_preferred_card(
    client: AsyncOpenAI,
    instruction: str,
    highlight: Highlight,
    cards: list[HighlightReference],
    params: SamplingParams,
    rng: random.Random,
) -> tuple[HighlightReference, str] | None:
    identified_cards = _randomly_identify_cards(cards, rng)
    ordered_cards = [f"**card_id={id}**:\n{card['content']}" for id, card in identified_cards.items()]
    rng.shuffle(ordered_cards)
    formatted_cards = "\n\n".join(ordered_cards)

    source_meta = highlight["source_meta"]
    assert isinstance(source_meta, dict), "Source meta is not a dictionary"

    interpretation = highlight["highlight_interpretation"]
    assert isinstance(interpretation, str), "Interpretation is not a string"

    user_prompt = _CHOOSE_FOUR_PROMPT_TEMPLATE.format(
        title=source_meta["title"],
        url=highlight["source_url"],
        author=source_meta["author"],
        highlight=format_highlight_text(highlight),
        highlight_interpretation=interpretation,
        cards=formatted_cards,
    )

    response = await asyncio.wait_for(
        client.chat.completions.create(
            model=params.model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_prompt},
            ],
            temperature=params.temperature,
            reasoning_effort=params.reasoning_effort,
            **params.extra_kwargs,
        ),
        timeout=60,  # 1 minute timeout
    )

    result_text = response.choices[0].message.content
    if result_text is None:
        return None

    match = re.search(_JUDGE_CARD_ID_REGEX, result_text)
    if match is None:
        return None
    card_id = match.group(1)
    if card_id is None:
        return None

    card = identified_cards.get(card_id)
    assert card is not None, "card_id not found in identified cards"
    return card, result_text


class _ChooseFourResult(TypedDict):
    highlight_id: int
    model: str
    targets: list[str]
    chosen_card: str
    chosen_task_type: str
    ground_truth_card_content: str
    correct: bool
    completion: str


def _has_valid_references(references: list[HighlightReference]) -> bool:
    # must have at least 3 references
    if len(references) < 3:
        return False

    construction_references = [card for card in references if card["tier"] != "T0"]
    if len(construction_references) < 3:
        return False

    # must have at least one pluckable card
    assert len([card for card in references if card["tier"] == "T3"]) >= 1, (
        "Must have at least one T3 reference"
    )

    return True


async def main(
    dataset: Dataset,
    client: AsyncOpenAI,
    instruction_file: str,
    params: SamplingParams,
    out_dir: str,
    max_concurrency: int = 10,
    batch_size: int = 20,
):
    with open(instruction_file, "r") as f:
        instruction = f.read()

    # Construct existing results path
    instruction_name = instruction_file.split("/")[-1].split(".")[0]
    effort = params.reasoning_effort if isinstance(params.reasoning_effort, str) else "none"
    results_path = os.path.join(out_dir, f"{instruction_name}_{params.model}_reasoning_{effort}.jsonl")

    # Load existing results or create empty DataFrame
    if not os.path.exists(results_path):
        keys = list(_ChooseFourResult.__annotations__.keys())
        df = pd.DataFrame(columns=keys)  # pyright: ignore[reportArgumentType]
    else:
        df = pd.read_json(results_path, lines=True)

    # Add result_id if it doesn't exist
    if "result_id" not in df.columns:
        df["result_id"] = range(len(df))

    # filter for tasks that have at least 3 rows
    dataset = dataset.filter(
        lambda x: _has_valid_references(x["references"]), desc="Filtering for valid references"
    )

    # Initialize DataFrame with all dataset highlights if empty
    if df.empty:
        logging.info("Initializing result rows for all highlights...")
        all_results = []
        for idx, highlight in enumerate(dataset):
            highlight = cast(Highlight, highlight)
            all_results.append(
                {
                    "result_id": idx,
                    "highlight_id": highlight["highlight_id"],
                    "model": params.model,
                    "targets": [],
                    "chosen_card": None,
                    "chosen_task_type": None,
                    "ground_truth_card_content": None,
                    "correct": None,
                }
            )
        df = pd.DataFrame(all_results)
        df.to_json(results_path, orient="records", lines=True)
        logging.info(f"Initialized {len(df)} result rows")

    df = df.set_index("result_id", drop=False)

    # Find pending evaluations (where chosen_card is None)
    pending_mask = df["chosen_card"].isna()
    pending_ids = df.loc[pending_mask, "result_id"].astype(int).tolist()

    if len(pending_ids) == 0:
        logging.info("No pending evaluations")
        return

    logging.info(f"Processing {len(pending_ids)} pending highlights")

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _process(result_id: int) -> tuple[int, dict[str, Any]]:
        # Get the highlight from the dataset
        highlight = cast(Highlight, dataset[result_id])

        # Create a deterministic RNG for this highlight to avoid async race conditions
        rng = random.Random(highlight["highlight_id"])

        # choose one card randomly and then see if the model can classify the task type
        refs = highlight["references"]
        pluckable_cards = [card for card in refs if card["tier"] == "T3"]
        assert len(pluckable_cards) > 0, "No pluckable cards found"

        non_pluckable_cards = [card for card in refs if card["tier"] in ["T1", "T2"]]
        assert len(non_pluckable_cards) > 0, "No non-pluckable cards found"

        pluckable_card = rng.choice(pluckable_cards)
        non_pluckable_cards = (
            non_pluckable_cards if len(non_pluckable_cards) <= 3 else rng.sample(non_pluckable_cards, 3)
        )

        targets = [pluckable_card] + non_pluckable_cards
        assert len(targets) <= 4 and len(targets) >= 2
        rng.shuffle(targets)

        async with semaphore:
            try:
                result = await choose_most_preferred_card(
                    client=client,
                    instruction=instruction,
                    highlight=highlight,
                    cards=targets,
                    params=params,
                    rng=rng,
                )
                if result is None:
                    logging.warning(f"No card chosen for highlight {highlight['highlight_id']}")
                    return result_id, {}
            except asyncio.TimeoutError:
                logging.error(f"Timeout processing highlight {highlight['highlight_id']}")
                return result_id, {}
            except Exception as e:
                logging.error(f"Error processing highlight {highlight['highlight_id']}: {e}")
                return result_id, {}

        chosen_card, result_text = result

        return result_id, {
            "targets": [card["content"] for card in targets],
            "chosen_card": chosen_card["content"],
            "chosen_task_type": chosen_card["task_type"],
            "ground_truth_card_content": pluckable_card["content"],
            "correct": chosen_card["id"] == pluckable_card["id"],
            "completion": result_text,
        }

    # Process in batches with checkpointing
    with tqdm(total=len(pending_ids), desc="Processing highlights") as pbar:
        for start in range(0, len(pending_ids), batch_size):
            batch_ids = pending_ids[start : start + batch_size]
            batch_results = await asyncio.gather(*(_process(result_id) for result_id in batch_ids))

            for result_id, update in batch_results:
                for k, v in update.items():
                    df.at[result_id, k] = v

            # Save checkpoint after each batch
            df.to_json(results_path, orient="records", lines=True)
            pbar.update(len(batch_ids))

    # Log summary statistics
    completed_mask = df["chosen_card"].notna()
    n_completed = completed_mask.sum()
    n_correct = df.loc[completed_mask, "correct"].sum()
    accuracy = n_correct / n_completed if n_completed > 0 else 0.0
    logging.info(f"Completed {n_completed} evaluations with {n_correct} correct ({accuracy:.2%} accuracy)")


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="laddermedia/srs-highlights")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--reasoning-effort", type=str, default=None, choices=["low", "medium", "high"])
    parser.add_argument("--instruction-file", type=str, required=True)
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of highlights to evaluate before saving checkpoint. Defaults to max_concurrency * 2.",
    )
    parser.add_argument("--base-url", type=str, default=None, help="Custom base URL for OpenAI API")
    parser.add_argument("--auth-token", type=str, default=None, help="Custom authentication token")
    parser.add_argument("--out-dir", type=str, default="./memory_machines/forced_choice/results")
    parser.add_argument(
        "--extra-body",
        type=str,
        default=None,
        help="JSON string for extra_body parameter (e.g., for Claude thinking)",
    )
    args = parser.parse_args()

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

    # Use Omit() if temperature not specified
    temperature = args.temperature if args.temperature is not None else Omit()
    reasoning_effort = args.reasoning_effort if args.reasoning_effort is not None else Omit()

    # Build extra_kwargs if extra_body is specified
    extra_kwargs = {}
    if args.extra_body is not None:
        extra_kwargs["extra_body"] = json.loads(args.extra_body)

    params = SamplingParams(
        model=args.model, temperature=temperature, reasoning_effort=reasoning_effort, extra_kwargs=extra_kwargs
    )
    logging.info(
        f"Sampling with model: {args.model}, temperature: {temperature}, reasoning_effort: {reasoning_effort}"
    )

    logging.info(f"Loading dataset from {args.dataset}")
    dataset = load_dataset(args.dataset, split="grounded")
    assert isinstance(dataset, Dataset), "Dataset is not a Dataset"

    # Create output directory if it doesn't exist
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    batch_size = args.batch_size if args.batch_size is not None else args.max_concurrency * 2

    asyncio.run(
        main(
            dataset=dataset,
            client=client,
            instruction_file=args.instruction_file,
            params=params,
            out_dir=args.out_dir,
            max_concurrency=args.max_concurrency,
            batch_size=batch_size,
        )
    )
