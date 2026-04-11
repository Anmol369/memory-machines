"""
Builds the srs-highlights dataset from the srs-prompts dataset by grouping prompts by highlight and selecting balanced grounded examples for task tiering evaluation.

Example usage:
    python -m memory_machines.highlight.build_dataset --dataset laddermedia/srs-prompts --push --repo-name laddermedia/srs-highlights
"""

import argparse
import logging
import random
from collections import Counter, defaultdict
from typing import cast

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from memory_machines.utils.tiers import get_tier_from_task_type
from memory_machines.utils.types import Highlight, HighlightReference, MemoryPrompt

MAX_ANNOTATIONS_PER_SOURCE = 6
TIER_ORDER = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}


def _build_reference(card: MemoryPrompt) -> HighlightReference:
    return {
        "id": card["id"],
        "content": card["content"],
        "task_type": card["task_type"],
        "tier": get_tier_from_task_type(card["task_type"]),
        "test_target": False,
    }


def _group_by_highlight_id(rows: Dataset) -> dict[int, list[MemoryPrompt]]:
    """Groups dataset rows by their highlight_id."""
    grouped: dict[int, list[MemoryPrompt]] = defaultdict(list)
    for row in rows:
        assert isinstance(row, dict), "Row is not a dict"
        assert "highlight_id" in row, "highlight_id is not in row"
        assert isinstance(row["highlight_id"], int), "highlight_id is not an integer"
        grouped[row["highlight_id"]].append(cast(MemoryPrompt, row))
    return grouped


def _is_well_posed_candidate(candidate_idx: int, cards: list[MemoryPrompt]) -> bool:
    """
    Check if selecting this candidate as target would result in a well-posed row.

    A candidate is well-posed if there's at least one OTHER card with tier >= candidate's tier.
    This ensures the model has grounding context for what the target tier (or higher) looks like.
    """
    candidate_tier = get_tier_from_task_type(cards[candidate_idx]["task_type"])
    candidate_rank = TIER_ORDER.get(candidate_tier, -1)

    for idx, card in enumerate(cards):
        if idx == candidate_idx:
            continue
        card_tier = get_tier_from_task_type(card["task_type"])
        if TIER_ORDER.get(card_tier, -1) >= candidate_rank:
            return True
    return False


def _select_target_for_balance(tier_counts: Counter[str], cards: list[MemoryPrompt]) -> int:
    """
    Selects a target card index that best balances the tier distribution.

    Only considers candidates that would result in a well-posed row (i.e., there's
    at least one other card with tier >= the candidate's tier).

    Prioritizes tiers with the lowest counts. Cards are shuffled to avoid length bias.
    Returns the index of the selected card in the original list.
    """
    assert len(cards) > 0, "No cards found"

    # Filter to only well-posed candidates
    well_posed_indices = [
        idx for idx in range(len(cards)) if _is_well_posed_candidate(idx, cards)
    ]

    # If no well-posed candidates, fall back to all cards
    if not well_posed_indices:
        well_posed_indices = list(range(len(cards)))

    # Order tiers from least to most common (prefer underrepresented tiers)
    tiers_by_priority = sorted(tier_counts.keys(), key=lambda t: tier_counts[t])

    # Shuffle to avoid length bias when multiple cards have same tier priority
    candidates = [(idx, cards[idx]) for idx in well_posed_indices]
    random.shuffle(candidates)

    best_index = None
    best_priority = float("inf")

    for original_idx, card in candidates:
        tier = get_tier_from_task_type(card["task_type"])
        priority = tiers_by_priority.index(tier)
        if priority < best_priority:
            best_priority = priority
            best_index = original_idx

    assert best_index is not None, "Best choice not found"
    return best_index


def _is_valid_highlight(cards: list[MemoryPrompt]) -> bool:
    """
    Checks if a highlight has valid cards for training.

    Requires:
    - At least 2 cards total
    - At least 1 pluckable card
    - At least 1 non-pluckable card that isn't off-target
    """
    if len(cards) < 2:
        return False

    has_pluckable = any(card["pluckable"] for card in cards)
    has_valid_non_pluckable = any(not card["pluckable"] and "off-target" not in card["tags"] for card in cards)

    return has_pluckable and has_valid_non_pluckable


def _balance_dataset(split: str, dataset: Dataset) -> list[Highlight]:
    """
    Balances the dataset by selecting grounded examples to even out tier distribution.

    Limits annotations per source to avoid overfitting.
    """
    tier_counts = Counter({tier: 0 for tier in ["T0", "T1", "T2", "T3"]})
    rows: list[Highlight] = []

    # Group annotations by source URL
    annotations_by_source: defaultdict[str, list[tuple[int, list[MemoryPrompt]]]] = defaultdict(list)
    for highlight_id, cards in _group_by_highlight_id(dataset).items():
        source_url = cards[0]["source_url"]
        annotations_by_source[source_url].append((highlight_id, cards))

    for source_url, annotations in annotations_by_source.items():
        # Identify valid highlights for this source
        valid_highlight_ids = {
            highlight_id for highlight_id, cards in annotations if _is_valid_highlight(cards)
        }

        # Limit highlights per source to avoid overfitting
        if len(valid_highlight_ids) > MAX_ANNOTATIONS_PER_SOURCE:
            selected_highlight_ids = set(random.sample(list(valid_highlight_ids), MAX_ANNOTATIONS_PER_SOURCE))
        else:
            selected_highlight_ids = valid_highlight_ids

        for highlight_id, cards in annotations:
            target_index = _select_target_for_balance(tier_counts, cards)
            target = cards[target_index]
            target_tier = get_tier_from_task_type(target["task_type"])

            references = [_build_reference(card) for card in cards]
            references[target_index]["test_target"] = True

            # Determine match type
            if highlight_id not in valid_highlight_ids:
                match_type = "underspecified"
            elif highlight_id not in selected_highlight_ids:
                match_type = "overflow"
            else:
                match_type = "grounded"

            # Only count grounded highlights toward distribution
            if match_type == "grounded":
                tier_counts[target_tier] += 1

            rows.append(
                {
                    "task_id": f"{highlight_id}-{target['id']}",
                    "split": split,
                    "match_type": match_type,
                    "highlight_id": highlight_id,
                    "source_url": target["source_url"],
                    "source_meta": target["source_meta"],
                    "highlight": target["highlight"],
                    "highlight_interpretation": target["highlight_interpretation"],
                    "references": references,
                }
            )

    return rows


def build_task_tiering_dataset(dataset: DatasetDict) -> DatasetDict:
    """Constructs a balanced task tiering dataset based on the srs-prompts dataset by evenly selecting task types."""
    rows = []
    # Sort split names for deterministic processing order
    for split in sorted(dataset.keys()):
        balanced_rows = _balance_dataset(split=str(split), dataset=dataset[split])
        rows.extend(balanced_rows)

    df = pd.DataFrame(rows)

    print("\n\nTier Distribution:")
    df_grounded = df[df["match_type"] == "grounded"].copy()

    def _get_target_tier(references: list[HighlightReference]) -> str:
        target = next(ref for ref in references if ref["test_target"])
        return get_tier_from_task_type(target["task_type"])

    df_grounded["target_tier"] = df_grounded["references"].apply(_get_target_tier)  # pyright: ignore[reportAttributeAccessIssue]
    tier_dist = df_grounded.groupby(["split", "target_tier"]).size()
    print(tier_dist)

    print("\n\nMatch Type Distribution:")
    eval_type_dist = df.groupby(["split", "match_type"]).size()
    print(eval_type_dist)

    dataset_splits: dict[str, Dataset] = {}
    for match_type, df_match_type in df.groupby("match_type"):
        # exlcude the match type column
        df_match_rows = df_match_type.drop(columns=["match_type"])
        dataset_splits[str(match_type)] = Dataset.from_list(df_match_rows.to_dict(orient="records"))

    print("Number of unique sources by match type:")
    print(df.groupby("match_type")["source_url"].nunique())

    key_ordering = ["grounded", "underspecified", "overflow"]
    assert set(dataset_splits.keys()) == set(key_ordering), "Dataset splits do not match key ordering"

    return DatasetDict({match_type: dataset_splits[match_type] for match_type in key_ordering})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="laddermedia/srs-prompts")
    parser.add_argument(
        "--push",
        action="store_true",
    )
    parser.add_argument("--repo-name", type=str, default="srs-highlights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seed for deterministic behavior
    random.seed(args.seed)

    dataset = load_dataset(args.dataset)
    assert isinstance(dataset, DatasetDict)

    task_tiering_dataset = build_task_tiering_dataset(dataset)
    print(task_tiering_dataset)

    # shuffle the dataset such that the HF page view is more random
    task_tiering_dataset = task_tiering_dataset.shuffle(seed=args.seed)

    if args.push:
        print(f"\nPushing dataset to Hugging Face Hub: {args.repo_name}")
        task_tiering_dataset.push_to_hub(args.repo_name)
        print("Done!")
