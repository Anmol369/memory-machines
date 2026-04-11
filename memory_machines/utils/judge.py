"""
LLM-as-a-judge utilities.

This module provides a small wrapper around the OpenAI Chat Completions API that can be
used to run *any* binary LLM judge, as long as the model can be instructed to emit a JSON
object containing a single boolean verdict.

## Judge contract (important)

`call_openai_judge()` is intentionally strict about what it parses:

- We request `response_format={"type": "json_object"}`.
- The model response must be a JSON object containing **exactly one boolean value**
  anywhere in the object. That boolean is interpreted as the verdict.
- Optionally, the object may include a `"reason"` field (string) for debugging.

Example accepted shapes:

- `{"pluckable": true}`
- `{"is_pluckable": false, "reason": "Off-target to the highlight."}`

If the model returns multiple booleans, parsing will fail with an assertion error.
This design keeps the evaluator robust against schema drift while remaining compatible
with slightly different key names.
"""

from dataclasses import dataclass, field
import json
from typing import Any, Literal, TypedDict
from openai import AsyncOpenAI, Omit
from openai.types.chat import ChatCompletion
from openai.types.completion_usage import CompletionUsage
from memory_machines.utils.format import format_user_message
from memory_machines.utils.types import MemoryPrompt


@dataclass
class SamplingParams:
    """Parameters controlling how the judge model is sampled."""

    model: str
    temperature: float | Omit
    reasoning_effort: Literal["low", "medium", "high"] | Omit
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "SamplingParams":
        """Create a copy of the sampling params."""
        return SamplingParams(
            model=self.model,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            extra_kwargs=self.extra_kwargs.copy(),
        )


class JudgeUsage(TypedDict):
    """Subset of usage information persisted alongside each judge verdict."""

    completion_tokens: int
    prompt_tokens: int
    thinking_tokens: int
    total_tokens: int


class JudgeResult(TypedDict):
    """Parsed judge output returned by `call_openai_judge()`."""

    prediction: bool
    reason: str | None
    usage: JudgeUsage


async def call_openai_judge(
    client: AsyncOpenAI, instruction: str, row: MemoryPrompt, params: SamplingParams
) -> JudgeResult | None:
    """
    Call a chat model as a judge and parse a strict boolean verdict.

    Inputs:
    - `instruction`: system prompt that explains the rubric and required JSON format.
    - `row`: a single structured example (this repo uses `MemoryPrompt` rows).
    - `params`: model/sampling configuration.

    Returns:
    - `JudgeResult` if a JSON response could be parsed
    - `None` if the model response had no content
    """

    # Detect if we're using Anthropic API. Special case since the Anthropic
    # API when using the Chat Completions API doesn't support the response_format parameter.
    is_anthropic = _is_anthropic_base_url(client)

    response: ChatCompletion = await client.chat.completions.create(
        model=params.model,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": format_user_message(row)},
        ],
        temperature=params.temperature,
        reasoning_effort=params.reasoning_effort,
        **({"response_format": {"type": "json_object"}} if not is_anthropic else {}),
        **params.extra_kwargs,
    )

    result_text = response.choices[0].message.content
    if result_text is None:
        return None

    # Parse JSON response (handles both fenced markdown and direct JSON)
    result_json = _parse_json_response(result_text)

    # extract the value which is a boolean
    boolean_values = [value for key, value in result_json.items() if isinstance(value, bool)]
    assert len(boolean_values) == 1, "Expected exactly one boolean value"
    judge_prediction = boolean_values[0]

    reason = result_json.get("reason", None)

    assert isinstance(judge_prediction, bool), "Judge prediction is not a boolean"
    assert reason is None or isinstance(reason, str), "Reason is not a string"
    assert response.usage is not None, "Usage is not None"
    assert isinstance(response.usage, CompletionUsage), "Usage is not a CompletionUsage"

    if response.usage.completion_tokens_details is not None:
        thinking_tokens = response.usage.completion_tokens_details.reasoning_tokens or 0
    else:
        thinking_tokens = 0

    judge_usage: JudgeUsage = {
        "completion_tokens": response.usage.completion_tokens,
        "prompt_tokens": response.usage.prompt_tokens,
        "thinking_tokens": thinking_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return {
        "prediction": judge_prediction,
        "reason": reason,
        "usage": judge_usage,
    }


def _is_anthropic_base_url(client: AsyncOpenAI) -> bool:
    """Check if the client is configured to use Anthropic's API."""
    base_url = str(client.base_url)
    return "anthropic.com" in base_url.lower()


def _parse_json_response(response_text: str) -> dict[str, Any]:
    """
    Parse JSON from a model response, handling both fenced markdown and direct JSON.

    Tries in order:
    1. Extract JSON from markdown code fence (```json ... ```)
    2. Parse directly as JSON

    Raises ValueError if neither approach succeeds.
    """
    response_text = response_text.strip()

    # Try extracting from markdown fence first
    if response_text.startswith("```json") and response_text.endswith("```"):
        json_text = response_text[7:-3].strip()
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            pass  # Fall through to direct parsing

    # Try parsing directly
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response_text}")
