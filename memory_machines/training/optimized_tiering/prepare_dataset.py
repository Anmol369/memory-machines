"""
Prepares the srs-highlights dataset for DSPy GEPA optimization.

Converts highlights with grounded references into DSPy Example format for
tier classification training and validation.

Example usage:
    python -m memory_machines.training.optimized_tiering.prepare_dataset
    python -m memory_machines.training.optimized_tiering.prepare_dataset --output-dir custom/path --seed 123
"""

import argparse
from collections import defaultdict
import logging
import pickle
import random
from pathlib import Path

import dspy
from datasets import Dataset, load_dataset

from memory_machines.tiering.classify import format_reference_cards, format_source_context
from memory_machines.utils.tiers import get_tier_from_task_type
from memory_machines.utils.types import Highlight, HighlightReference


def prepare_dspy_dataset(
    dataset_name: str = "laddermedia/srs-highlights",
    split_name: str = "grounded",
    train_val_split: float = 0.5,
    output_dir: str = "memory_machines/training/data/optimized_tiering",
    seed: int = 42,
):
    """
    Load srs-highlights dataset and convert to DSPy Example format.

    Args:
        dataset_name: HuggingFace dataset name
        split_name: Split to use (default: "grounded")
        train_val_split: Fraction of data to use for training (rest goes to validation)
        output_dir: Directory to save prepared dataset
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    logging.info(f"Loading dataset: {dataset_name} (split: {split_name})")
    dataset = load_dataset(dataset_name, split=split_name)
    assert isinstance(dataset, Dataset)
    logging.info(f"Total highlights in {split_name} split: {len(dataset)}")
    dataset_list = list(dataset)

    examples = []

    for row in dataset_list:
        highlight = Highlight(**row)  # type: ignore
        references = highlight["references"]

        # Find the test target (the prompt we're trying to predict the tier for)
        test_targets = [ref for ref in references if ref["test_target"]]
        if len(test_targets) != 1:
            logging.warning(
                f"Highlight {highlight.get('id', 'unknown')} has {len(test_targets)} test targets, skipping"
            )
            continue

        assert len(test_targets) == 1, "Expected exactly one test target"
        test_target = test_targets[0]

        # Get other references (for grounding/calibration)
        reference_cards: list[HighlightReference] = [ref for ref in references if not ref["test_target"]]

        # Skip highlights without reference cards (can't do grounded training)
        if len(reference_cards) == 0:
            logging.debug(f"Highlight {highlight.get('id', 'unknown')} has no reference cards, skipping")
            continue

        # Format source context and reference cards using existing functions
        source_description = format_source_context(highlight)
        reference_cards_formatted = format_reference_cards(reference_cards)
        target_prompt = test_target["content"]
        correct_tier = get_tier_from_task_type(test_target["task_type"])

        # Create DSPy Example
        example = dspy.Example(
            {
                "source_description": source_description,
                "reference_cards": reference_cards_formatted,
                "target_prompt": target_prompt,
                "correct_tier": correct_tier,
            }
        ).with_inputs("source_description", "reference_cards", "target_prompt")

        examples.append(example)

    logging.info(f"Total examples created: {len(examples)}")

    # Calculate tier distribution
    tier_counts: dict[str, int] = defaultdict(int)
    for example in examples:
        tier_counts[example.correct_tier] += 1

    logging.info("Tier distribution:")
    for tier in ["T0", "T1", "T2", "T3"]:
        count = tier_counts[tier]
        pct = (count / len(examples) * 100) if len(examples) > 0 else 0
        logging.info(f"  {tier}: {count} ({pct:.1f}%)")

    # Shuffle and split into train/val
    random.shuffle(examples)
    split_point = int(len(examples) * train_val_split)
    train_set = examples[:split_point]
    val_set = examples[split_point:]

    logging.info("\nSplit sizes:")
    logging.info(f"  Train: {len(train_set)}")
    logging.info(f"  Val: {len(val_set)}")

    # Save datasets
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train_set.pkl"
    val_path = output_path / "val_set.pkl"

    with open(train_path, "wb") as f:
        pickle.dump(train_set, f)
    with open(val_path, "wb") as f:
        pickle.dump(val_set, f)

    logging.info(f"\nDataset saved to: {output_dir}")
    logging.info(f"  Train: {train_path}")
    logging.info(f"  Val: {val_path}")
    logging.info("Dataset preparation complete!")

    return train_set, val_set


def load_prepared_dataset(data_dir: str) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Load prepared DSPy dataset from disk."""
    train_path = Path(data_dir) / "train_set.pkl"
    val_path = Path(data_dir) / "val_set.pkl"

    with open(train_path, "rb") as f:
        train_set = pickle.load(f)
    with open(val_path, "rb") as f:
        val_set = pickle.load(f)

    return train_set, val_set


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Prepare srs-highlights dataset for DSPy GEPA optimization")
    parser.add_argument(
        "--dataset",
        type=str,
        default="laddermedia/srs-highlights",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="grounded",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--train-val-split",
        type=float,
        default=0.5,
        help="Fraction of data to use for training (rest goes to validation)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="memory_machines/training/data/optimized_tiering",
        help="Output directory for prepared dataset",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    prepare_dspy_dataset(
        dataset_name=args.dataset,
        split_name=args.split,
        train_val_split=args.train_val_split,
        output_dir=args.output_dir,
        seed=args.seed,
    )
