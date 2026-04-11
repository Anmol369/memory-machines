"""
Prepares the srs-highlights dataset for SFT tiering classification training.

Creates 4-class tier labels (T0, T1, T2, T3) with hardcoded tier definitions in system prompt.
Formats data as chat messages for TRL SFTTrainer with assistant-only loss.

Example usage:
    python -m memory_machines.training.sft_tiering.prepare_dataset
    python -m memory_machines.training.sft_tiering.prepare_dataset --max-prompts-per-source 10 --seed 123
"""

import argparse
import logging
import random
from collections import defaultdict
from typing import cast

from datasets import Dataset, DatasetDict, load_dataset

from memory_machines.tiering.classify import format_reference_cards, format_source_context
from memory_machines.utils.types import Highlight, MemoryPromptTier

# Hardcoded system prompt with tier definitions
TIER_SYSTEM_PROMPT = """You are an expert at evaluating memory prompts for spaced repetition systems.

Given a user's highlight from a document and a candidate memory prompt, classify it into one of four tiers:

**T0 (off-target)**: The prompt does not capture what the user highlighted or is interested in. It focuses on details not aligned with the user's likely intent.

**T1 (needs-refactor)**: The prompt is in the right region of meaning but structurally ineffective for review. The question construction fails reviewability tests: too vague, lacks sufficient contextual cues, or is too generic.

**T2 (needs-polish)**: The prompt is semantically aligned with the user's interest and reviewable, but needs minor wording improvements. The core targeting and construction are sound.

**T3 (ready-to-review)**: The prompt satisfies both criteria: (1) well-targeted to the user's interest, AND (2) well-constructed for spaced repetition review.

Respond with exactly one tier label: T0, T1, T2, or T3."""

# User prompt templates for tier classification (grounded and ungrounded)
USER_PROMPT_GROUNDED_TEMPLATE = """{source_description}

Here are other memory prompts for this same highlight, rated by tier (T3 is best, T0 is worst):

{reference_cards}

Now, evaluate this candidate memory prompt:
{content}

What tier does this memory prompt belong to?"""


def prepare_sft_tiering_dataset(
    dataset_name: str = "laddermedia/srs-highlights",
    output_dir: str = "memory_machines/training/data/sft_tiering",
    seed: int = 42,
):
    """
    Load srs-highlights, apply source-level limiting, create chat messages, and save dataset.

    Args:
        dataset_name: HuggingFace dataset name
        max_prompts_per_source: Maximum number of prompts per source URL
        output_dir: Directory to save prepared dataset
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    logging.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="grounded")
    assert isinstance(dataset, Dataset), "Dataset is not a DatasetDict"

    # Apply limiting and create chat messages
    prepared_splits: dict[str, list[dict]] = defaultdict(list)
    sources_exceeding_limit = 0

    # Create chat message for each row
    for highlight in dataset:
        highlight = cast(Highlight, highlight)

        # Get tier from task_type
        target_mask = next(ref for ref in highlight["references"] if ref["test_target"])

        source_description = format_source_context(highlight)

        assert len(highlight["references"]) > 0, "Expected at least one reference card"
        reference_cards_formatted = format_reference_cards(
            [ref for ref in highlight["references"] if not ref["test_target"]]
        )
        user_message = USER_PROMPT_GROUNDED_TEMPLATE.format(
            source_description=source_description,
            reference_cards=reference_cards_formatted,
            content=target_mask["content"],
        )

        # Create chat conversation with system, user, and assistant messages
        conversation = [
            {"role": "system", "content": TIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": target_mask["tier"]},  # Just "T0", "T1", "T2", or "T3"
        ]

        prepared_splits[highlight["split"]].append(
            {
                "id": f"{highlight['highlight_id']}-{target_mask['id']}",
                "messages": conversation,
                "tier": target_mask["tier"],  # Ground truth for evaluation
                "source_url": highlight["source_url"],  # For metadata
            }
        )

    total_limited_rows = sum(len(rows) for rows in prepared_splits.values())
    logging.info(f"  Samples after limiting: {total_limited_rows}")
    logging.info(f"  Sources exceeding limit: {sources_exceeding_limit}")

    # Calculate tier distribution
    tier_counts = defaultdict(int)
    for row in [row for rows in prepared_splits.values() for row in rows]:
        tier_counts[row["tier"]] += 1

    logging.info("  Tier distribution:")
    for tier in MemoryPromptTier.__args__:
        count = tier_counts[tier]
        pct = count / total_limited_rows * 100 if total_limited_rows > 0 else 0.0
        logging.info(f"    {tier}: {count} ({pct:.1f}%)")

    # Create dataset from rows
    prepared_dataset = DatasetDict({split: Dataset.from_list(rows) for split, rows in prepared_splits.items()})

    logging.info(f"\nSaving dataset to: {output_dir}")
    prepared_dataset.save_to_disk(output_dir)
    logging.info("Dataset preparation complete!")
    logging.info(f"\nFinal dataset: {prepared_dataset}")

    return prepared_dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Prepare srs-highlights dataset for SFT tiering classification"
    )
    parser.add_argument(
        "--dataset", type=str, default="laddermedia/srs-highlights", help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--max-prompts-per-source",
        type=int,
        default=12,
        help="Maximum number of prompts per source (for overfitting prevention)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="memory_machines/training/data/sft_tiering",
        help="Output directory for prepared dataset",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    prepare_sft_tiering_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        seed=args.seed,
    )
