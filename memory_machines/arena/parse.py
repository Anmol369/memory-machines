"""
Parsing utilities for arena memory prompt generation.
"""

import re

_MEMORY_PROMPT_REGEX = r"(Q[.:]\s*.+?\s*A[.:]\s*.+?)(?=\s*(?:```|---|Q[.:]|$))"


def parse_memory_prompts(result_text: str) -> list[str]:
    """Parse Q/A memory prompts from model response.

    Args:
        result_text: Raw text from model completion

    Returns:
        List of parsed memory prompt strings
    """
    result_text = result_text.strip()
    memory_prompts = re.findall(_MEMORY_PROMPT_REGEX, result_text, re.DOTALL)
    return [mp.strip() for mp in memory_prompts]
