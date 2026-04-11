"""
Prepares the srs-prompts dataset for reward model training.

Creates preference pairs from tier hierarchy: T3 > T2 > T1 > T0
Formats pairs as ChatML messages for TRL RewardTrainer.

Example usage:
    python -m memory_machines.training.reward_model.prepare_dataset
    python -m memory_machines.training.reward_model.prepare_dataset --max-prompts-per-source 10 --seed 123
"""

import argparse
import logging
import random
from collections import defaultdict
from typing import cast

from datasets import Dataset, DatasetDict, load_dataset

from memory_machines.utils.format import format_highlight_text
from memory_machines.utils.types import MemoryPrompt
from memory_machines.utils.tiers import get_tier_from_task_type

# Tier ordering for preference pairs (higher is better)
TIER_ORDER = {"T3": 3, "T2": 2, "T1": 1, "T0": 0}

_user_prompt_template = """I am reading [{title}]({url}) by {author}

I highlighted the following text:
> {highlight}

Interpretation of the highlight within the context of the document:
{highlight_interpretation}


Generate a memory prompt.
"""


def prepare_reward_dataset(
    dataset_name: str = "laddermedia/srs-prompts",
    max_prompts_per_source: int = 12,
    output_dir: str = "memory_machines/training/data/reward_model",
    seed: int = 42,
):
    """
    Load srs-prompts, apply source-level limiting, create preference pairs, and save dataset.

    Args:
        dataset_name: HuggingFace dataset name
        max_prompts_per_source: Maximum number of prompts per source URL
        output_dir: Directory to save prepared dataset
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    logging.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    assert isinstance(dataset, DatasetDict), "Dataset is not a DatasetDict"

    prepared_splits = {}

    for split_name in dataset.keys():
        logging.info(f"\nProcessing split: {split_name}")
        split_data = dataset[split_name]

        # Group samples by source_url
        sources = defaultdict(list)
        for row in split_data:
            assert isinstance(row, dict), "Row is not a dictionary"
            sources[row["source_url"]].append(row)

        logging.info(f"  Total samples before limiting: {len(split_data)}")
        logging.info(f"  Unique sources: {len(sources)}")

        # Apply limiting and create preference pairs
        all_pairs = []
        sources_exceeding_limit = 0

        for source_url, rows in sources.items():
            # Limit to max_prompts_per_source
            if len(rows) > max_prompts_per_source:
                rows = random.sample(rows, max_prompts_per_source)
                sources_exceeding_limit += 1

            # Group by highlight_id to create pairs within same highlight
            highlights = defaultdict(list)
            for row in rows:
                highlights[row["highlight_id"]].append(row)

            # Create pairwise comparisons for each highlight
            for highlight_id, prompts in highlights.items():
                # Need at least 2 prompts with different tiers to create pairs
                if len(prompts) < 2:
                    continue

                # Generate all valid pairs (chosen > rejected)
                for i, prompt_i in enumerate(prompts):
                    tier_i = get_tier_from_task_type(prompt_i["task_type"])
                    prompt_i = cast(MemoryPrompt, prompt_i)

                    highlight_user_message = _user_prompt_template.format(
                        title=prompt_i["source_meta"]["title"],
                        url=prompt_i["source_url"],
                        author=prompt_i["source_meta"]["author"],
                        highlight=format_highlight_text(prompt_i),
                        highlight_interpretation=prompt_i["highlight_interpretation"],
                    )

                    for j, prompt_j in enumerate(prompts):
                        if i == j:
                            continue

                        tier_j = get_tier_from_task_type(prompt_j["task_type"])

                        # Only create pair if there's a clear preference
                        if TIER_ORDER[tier_i] > TIER_ORDER[tier_j]:
                            # tier_i is better (chosen), tier_j is worse (rejected)
                            all_pairs.append(
                                {
                                    "chosen": [
                                        {
                                            "role": "user",
                                            "content": highlight_user_message,
                                        },
                                        {"role": "assistant", "content": prompt_i["content"]},
                                    ],
                                    "rejected": [
                                        {"role": "user", "content": highlight_user_message},
                                        {"role": "assistant", "content": prompt_j["content"]},
                                    ],
                                    "chosen_tier": tier_i,
                                    "rejected_tier": tier_j,
                                    "highlight_id": highlight_id,
                                    "source_url": source_url,
                                }
                            )

        logging.info(f"  Sources exceeding limit: {sources_exceeding_limit}")
        logging.info(f"  Preference pairs generated: {len(all_pairs)}")

        # Calculate pair distribution
        pair_types = defaultdict(int)
        for pair in all_pairs:
            pair_type = f"{pair['chosen_tier']}v{pair['rejected_tier']}"
            pair_types[pair_type] += 1

        logging.info("  Pair distribution:")
        for pair_type in sorted(pair_types.keys(), key=lambda x: pair_types[x], reverse=True):
            count = pair_types[pair_type]
            pct = count / len(all_pairs) * 100
            logging.info(f"    {pair_type}: {count} ({pct:.1f}%)")

        # Create dataset from pairs
        prepared_splits[split_name] = Dataset.from_list(all_pairs)

    # Create DatasetDict and save
    prepared_dataset = DatasetDict(prepared_splits)
    logging.info(f"\nSaving dataset to: {output_dir}")
    prepared_dataset.save_to_disk(output_dir)
    logging.info("Dataset preparation complete!")
    logging.info(f"\nFinal dataset: {prepared_dataset}")

    return prepared_dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Prepare srs-prompts dataset for reward model training")
    parser.add_argument("--dataset", type=str, default="laddermedia/srs-prompts", help="HuggingFace dataset name")
    parser.add_argument(
        "--max-prompts-per-source",
        type=int,
        default=12,
        help="Maximum number of prompts per source (for overfitting prevention)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="memory_machines/training/data/reward_model",
        help="Output directory for prepared dataset",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    prepare_reward_dataset(
        dataset_name=args.dataset,
        max_prompts_per_source=args.max_prompts_per_source,
        output_dir=args.output_dir,
        seed=args.seed,
    )
