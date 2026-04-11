"""
Tier classification: Multi-class classification (T0-T3) of memory prompts for spaced
repetition systems. Evaluates whether generated memory prompts are production-ready,
need polish, need refactoring, or are off-target.
"""

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import cast

from openai import AsyncOpenAI

from memory_machines.utils.format import format_highlight_text
from memory_machines.utils.judge import SamplingParams
from memory_machines.utils.tiers import get_tier_from_task_type
from memory_machines.utils.types import Highlight, HighlightReference, MemoryPromptTier

_RESPONSE_TEMPLATE = r"\*\*Classification: (T\d)\*\*"


@lru_cache(maxsize=1)
def _load_instructions() -> tuple[str, str]:
    """Load grounded and ungrounded instruction templates from files."""
    module_dir = Path(__file__).resolve().parent
    instruction_grounded_path = module_dir / "instructions" / "rate_grounded.txt"
    instruction_ungrounded_path = module_dir / "instructions" / "rate_ungrounded.txt"

    with open(instruction_grounded_path, "r") as f:
        instruction_grounded = f.read()
    with open(instruction_ungrounded_path, "r") as f:
        instruction_ungrounded = f.read()

    return instruction_grounded, instruction_ungrounded


_source_context_template = """[{title}]({url}) by {author}

**Highlight:**
> {highlight}


**Interpretation of the highlight within the context of the document:**
{highlight_interpretation}
"""


def format_source_context(highlight: Highlight) -> str:
    """Format the source context for the tier classification prompt."""
    source_meta = highlight["source_meta"]
    assert isinstance(source_meta, dict), "Source meta is not a dictionary"

    assert "author" in source_meta, "Author is not in source meta"
    assert "title" in source_meta, "Title is not in source meta"

    url = highlight["source_url"]
    assert isinstance(url, str), "URL is not a string"

    return _source_context_template.format(
        title=source_meta["title"],
        url=url,
        author=source_meta["author"],
        highlight=format_highlight_text(highlight),
        highlight_interpretation=highlight["highlight_interpretation"],
    )


_rated_memory_prompts_template = """**Memory Prompt ({TIER})**:
{CONTENT}"""

_TIERS_ORDERING: list[MemoryPromptTier] = ["T3", "T2", "T1", "T0"]


def format_reference_cards(cards: list[HighlightReference]) -> str:
    """Format reference cards for grounded tier classification."""
    formatted_rated_memory_prompts = [
        (
            get_tier_from_task_type(mp["task_type"]),
            _rated_memory_prompts_template.format(
                TIER=get_tier_from_task_type(mp["task_type"]),
                CONTENT=mp["content"],
            ),
        )
        for mp in cards
    ]

    formatted_rated_memory_prompts.sort(key=lambda x: _TIERS_ORDERING.index(x[0]))
    formatted_rated_memory_prompts = reversed(formatted_rated_memory_prompts)

    return "\n\n".join([content.strip() for _, content in formatted_rated_memory_prompts])


async def _classify(
    client: AsyncOpenAI, user_message: str, params: SamplingParams
) -> tuple[MemoryPromptTier, str]:
    """
    Call the model to classify a memory prompt into a tier (T0-T3).

    Returns:
        tuple of (tier, full_completion_text)
    """
    response = await client.chat.completions.create(
        model=params.model,
        messages=[{"role": "user", "content": user_message}],
        temperature=params.temperature,
        reasoning_effort=params.reasoning_effort,
        **params.extra_kwargs,
    )
    logging.info(f"Response: {response.usage}")
    result = response.choices[0].message.content
    if result is None:
        raise Exception("Empty response from API")
    match = re.search(_RESPONSE_TEMPLATE, result)
    if match is None:
        raise Exception(f"Failed to find classification in response: {result}")
    tier = match.group(1)
    # Validate tier is in expected range
    if tier not in ["T0", "T1", "T2", "T3"]:
        raise Exception(f"Invalid classification: {tier}")
    return cast(MemoryPromptTier, tier), result


async def _classify_trained_model(
    client: AsyncOpenAI,
    highlight: Highlight,
    references: list[HighlightReference],
    target: str,
    params: SamplingParams,
) -> tuple[MemoryPromptTier, str]:
    """
    Call the model to classify a memory prompt into a tier (T0-T3) using a trained model.
    """
    from memory_machines.training.sft_tiering.prepare_dataset import (
        TIER_SYSTEM_PROMPT,
        USER_PROMPT_GROUNDED_TEMPLATE,
    )

    response_template = r"(T\d)"

    user_message = USER_PROMPT_GROUNDED_TEMPLATE.format(
        source_description=format_source_context(highlight),
        reference_cards=format_reference_cards(references),
        content=target,
    )
    response = await client.chat.completions.create(
        model=params.model,
        messages=[
            {"role": "system", "content": TIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=params.temperature,
        reasoning_effort=params.reasoning_effort,
        **params.extra_kwargs,
    )
    logging.info(f"Response: {response.usage}")
    result = response.choices[0].message.content
    if result is None:
        raise Exception("Empty response from API")
    match = re.search(response_template, result)
    if match is None:
        raise Exception(f"Failed to find classification in response: {result}")
    tier = match.group(1)
    # Validate tier is in expected range
    if tier not in ["T0", "T1", "T2", "T3"]:
        raise Exception(f"Invalid classification: {tier}")
    return cast(MemoryPromptTier, tier), result


async def classify_tier(
    client: AsyncOpenAI,
    highlight: Highlight,
    references: list[HighlightReference],
    target: str,
    params: SamplingParams,
) -> tuple[MemoryPromptTier, str]:
    """
    Classify a single memory prompt into a tier (T0-T3).

    Returns:
        tuple of (tier, completion_text)
    """

    if params.extra_kwargs.get("use_train_prompt", False):
        # special case for trained model (need to use the same system prompt as the training data)
        mutable_params = params.copy()
        mutable_params.extra_kwargs.pop("use_train_prompt")
        return await _classify_trained_model(
            client=client,
            highlight=highlight,
            references=references,
            target=target,
            params=mutable_params,
        )

    instruction_grounded, instruction_ungrounded = _load_instructions()

    if len(references) > 0:
        user_message = instruction_grounded.format(
            SOURCE_DESCRIPTION=format_source_context(highlight),
            RATED_MEMORY_PROMPTS=format_reference_cards(references),
            GENERATED_PROMPT=target,
        )
    else:
        user_message = instruction_ungrounded.format(
            SOURCE_DESCRIPTION=format_source_context(highlight),
            GENERATED_PROMPT=target,
        )

    return await _classify(client=client, user_message=user_message, params=params)
