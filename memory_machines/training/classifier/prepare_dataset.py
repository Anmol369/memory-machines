"""
Prepares the srs-prompts dataset for pluckability classification training.

Applies source-level limiting (12 prompts per source) and creates binary labels:
- Label 0: T0 (off-target) + T1 (needs-refactor) = unpluckable
- Label 1: T2 (needs-polish) + T3 (ready-to-review) = pluckable

Example usage:
    python -m memory_machines.training.classifier.prepare_dataset
    python -m memory_machines.training.classifier.prepare_dataset --max-prompts-per-source 10 --seed 123
"""

import argparse
import logging
import random
from collections import defaultdict

from datasets import Dataset, DatasetDict, load_dataset


def prepare_pluckability_dataset(
    dataset_name: str = "laddermedia/srs-prompts",
    max_prompts_per_source: int = 12,
    output_dir: str = "memory_machines/training/data/classifier",
    seed: int = 42,
):
    """
    Load srs-prompts, apply source-level limiting, create binary labels, and save dataset.

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
            sources[row["source_url"]].append(row)

        logging.info(f"  Total samples before limiting: {len(split_data)}")
        logging.info(f"  Unique sources: {len(sources)}")

        # Apply limiting and create binary labels
        limited_rows = []
        sources_exceeding_limit = 0

        for source_url, rows in sources.items():
            # Limit to max_prompts_per_source
            if len(rows) > max_prompts_per_source:
                rows = random.sample(rows, max_prompts_per_source)
                sources_exceeding_limit += 1

            # Create binary labels and keep essential fields
            for row in rows:
                # Binary label from pluckable field
                label = 1 if row["pluckable"] else 0

                limited_rows.append(
                    {
                        "id": row["id"],
                        "source_url": row["source_url"],
                        "source_meta": row["source_meta"],
                        "highlight": row["highlight"],
                        "highlight_interpretation": row["highlight_interpretation"],
                        "content": row["content"],
                        "task_type": row["task_type"],
                        "pluckable": row["pluckable"],
                        "label": label,
                    }
                )

        logging.info(f"  Samples after limiting: {len(limited_rows)}")
        logging.info(f"  Sources exceeding limit: {sources_exceeding_limit}")

        # Calculate class balance
        pluckable_count = sum(1 for row in limited_rows if row["label"] == 1)
        unpluckable_count = len(limited_rows) - pluckable_count
        logging.info("  Class balance (before balancing):")
        logging.info(
            f"    Unpluckable (label=0): {unpluckable_count} ({unpluckable_count / len(limited_rows) * 100:.1f}%)"
        )
        logging.info(
            f"    Pluckable (label=1): {pluckable_count} ({pluckable_count / len(limited_rows) * 100:.1f}%)"
        )

        # Balance classes for train split only (undersample majority class)
        if split_name == "train":
            pluckable_rows = [row for row in limited_rows if row["label"] == 1]
            unpluckable_rows = [row for row in limited_rows if row["label"] == 0]

            min_class_size = min(len(pluckable_rows), len(unpluckable_rows))

            if len(pluckable_rows) > min_class_size:
                pluckable_rows = random.sample(pluckable_rows, min_class_size)
            if len(unpluckable_rows) > min_class_size:
                unpluckable_rows = random.sample(unpluckable_rows, min_class_size)

            limited_rows = pluckable_rows + unpluckable_rows
            random.shuffle(limited_rows)

            logging.info(f"  Samples after balancing: {len(limited_rows)}")
            logging.info("  Class balance (after balancing):")
            logging.info(f"    Unpluckable (label=0): {len(unpluckable_rows)} (50.0%)")
            logging.info(f"    Pluckable (label=1): {len(pluckable_rows)} (50.0%)")

        # Create dataset from rows
        prepared_splits[split_name] = Dataset.from_list(limited_rows)

    # Create DatasetDict and save
    prepared_dataset = DatasetDict(prepared_splits)
    logging.info(f"\nSaving dataset to: {output_dir}")
    prepared_dataset.save_to_disk(output_dir)
    logging.info("Dataset preparation complete!")
    logging.info(f"\nFinal dataset: {prepared_dataset}")

    return prepared_dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Prepare srs-prompts dataset for pluckability classification")
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
        default="memory_machines/training/data/classifier",
        help="Output directory for prepared dataset",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    prepare_pluckability_dataset(
        dataset_name=args.dataset,
        max_prompts_per_source=args.max_prompts_per_source,
        output_dir=args.output_dir,
        seed=args.seed,
    )
