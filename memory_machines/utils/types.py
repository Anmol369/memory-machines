"""
Types for the srs-prompts and srs-highlights datasets used throughout this repo.
"""

from typing import Any, Literal, TypedDict

# 4-way human rating tier (T0–T3), represented as a string label.
#
# - Tier 3 (T3): suitable for SRS review as-is.
# - Tier 2 (T2): semantically aligned; needs minor polish but is still reviewable.
# - Tier 1 (T1): roughly in the right region of meaning, but structurally ineffective for review.
# - Tier 0 (T0): off-target; does not capture the user's intended target of recall / interest.
MemoryPromptTaskType = Literal["ready-to-review", "needs-polish", "needs-refactor", "off-target"]


class MemoryPrompt(TypedDict):
    """
    A single memory-prompt record from the srs-prompts dataset.

    The dataset is organized around highlights ("annotations") from source documents, and each
    highlight may have multiple candidate memory prompts (some good, some bad).
    """

    # Stable identifier for this specific memory prompt / "flashcard" row.
    id: int

    # Identifier for the underlying highlight/annotation this card was generated from.
    # Many cards can share the same `highlight_id`.
    highlight_id: int

    # Canonical URL for the source document the highlight came from.
    source_url: str

    # Additional source metadata (shape varies by dataset release; e.g. title/author/etc.).
    source_meta: dict[str, Any]

    # 4-way human rating tier for this memory prompt (T0–T3), represented as a string label.
    task_type: MemoryPromptTaskType

    # The exact highlighted span (often a narrow excerpt that may not stand alone).
    highlight: str

    # A model-produced interpretation of what the highlight indicates in the context of the full
    # source; used to reduce the "large context" dependency (may be incorrect).
    highlight_interpretation: str

    # The memory prompt content. Typically Q/A-formatted text intended for spaced-repetition review.
    content: str

    # Binary accept/reject signal used in parts of the essay/eval:
    # pluckable=True  for "needs-polish" or "ready-to-review"
    # pluckable=False for "needs-refactor" or "off-target"
    pluckable: bool

    # Free-form categorical tags (e.g. dataset bookkeeping, heuristics, or failure-mode labels).
    tags: list[str]


# Numeric tier labels (T0–T3) corresponding to the 4-way human rating scheme.
#
# - T3: ready-to-review (suitable for SRS review as-is)
# - T2: needs-polish (semantically aligned; needs minor edits but reviewable)
# - T1: needs-refactor (roughly on-topic but structurally ineffective)
# - T0: off-target (does not capture user's intended target of recall)
MemoryPromptTier = Literal["T0", "T1", "T2", "T3"]


class HighlightReference(TypedDict):
    """
    A single reference memory prompt used in the srs-highlights dataset.

    In the srs-highlights evaluation framework, each highlight is associated with multiple
    reference memory prompts of varying quality (typically T0-T3). These references serve
    two purposes:

    1. Ground-truth examples for highlight-conditioned classification - models are trained
       to predict which tier a given memory prompt belongs to based on the highlight and
       the reference prompts.

    2. Test grounded examples for evaluating judge reliability - one reference per highlight is marked
       as the "test_target", representing what a well-calibrated evaluator should select
       as the best prompt for that highlight.
    """

    # Stable identifier for the original memory prompt from srs-prompts.
    id: int

    # The memory prompt content (typically Q/A-formatted text).
    content: str

    # The 4-way human rating tier for this reference (T0–T3).
    task_type: MemoryPromptTaskType

    # String representation of the tier (e.g. "T0", "T1", "T2", "T3").
    tier: MemoryPromptTier

    # Boolean indicating whether this reference is the ground-truth "best" prompt for this
    # highlight. Exactly one reference per highlight is marked as the test target. This is
    # used to evaluate whether judges can reliably identify high-quality prompts when
    # comparing model generations to references.
    test_target: bool


class Highlight(TypedDict):
    """
    A single highlight record from the srs-highlights dataset. srs-highlights is derived from
    srs-prompts. Each `Highlight` represents one highlight from srs-prompts along with all its
    associated memory prompts as references.
    """

    # Unique identifier for this task (format: "{highlight_id}-{grounded_reference_id}").
    task_id: str

    # Identifier for the underlying highlight/annotation this task was generated from.
    highlight_id: int

    # The assigned split of the highlight in the original srs-prompts dataset (e.g. "train", "test").
    split: str

    # Canonical URL for the source document the highlight came from.
    source_url: str

    # Additional source metadata (shape varies; e.g. title, author, publication date).
    source_meta: dict

    # The exact highlighted span from the source document.
    highlight: str

    # A model-produced interpretation of what the highlight indicates in context.
    # Used to reduce dependency on full source text during evaluation.
    highlight_interpretation: str

    # List of reference memory prompts for this highlight, spanning multiple quality tiers.
    # Exactly one reference has `test_target=True`, representing the ground-truth best prompt.
    references: list[HighlightReference]
