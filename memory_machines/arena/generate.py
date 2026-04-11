import argparse
import asyncio
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import urllib.parse
from datasets import Dataset
import pandas as pd
from typing import TypedDict, cast
from openai import AsyncOpenAI, Omit
from datasets import load_dataset
from tqdm.asyncio import tqdm
from memory_machines.arena.parse import parse_memory_prompts
from memory_machines.tiering.classify import classify_tier
from memory_machines.utils.format import format_highlight_text
from memory_machines.utils.judge import SamplingParams
from memory_machines.utils.types import Highlight, MemoryPromptTier

_GENERATION_FULL_TEXT_REQUEST_TEMPLATE = """I am reading [{title}]({url}) by {author}

{full_text}

----

**Highlight:**
I highlighted the following text:
> {highlight}

Generate memory prompt(s) for my memory system based on what I found interesting in the highlight.
"""

_GENERATION_INTERPRETATION_REQUEST_TEMPLATE = """I am reading [{title}]({url}) by {author}

**Highlight:**
I highlighted the following text:
> {highlight}

**Interpretation of the highlight within the context of the document:**
{highlight_interpretation}

Generate memory prompt(s) for my memory system based on what I found interesting in the highlight.
"""


def _get_full_text(row: Highlight) -> str | None:
    parsed_url = urllib.parse.urlparse(row["source_url"])
    hostname = parsed_url.netloc
    path = parsed_url.path.strip("/")

    # Replace special characters with underscores
    hostname = re.sub(r"[^\w\-]", "_", hostname)
    path = re.sub(r"[^\w\-/]", "_", path)

    # Replace slashes in path with underscores for folder structure
    path = path.replace("/", "_")
    if not path:
        return None

    repo_path = os.path.join("repo", hostname, path)
    chunks_path = os.path.join(repo_path, "chunks.json")

    if not os.path.exists(chunks_path):
        return None

    try:
        with open(chunks_path, "r") as f:
            chunks = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"Error loading chunks for row {row['task_id']}: {e}")
        return None

    if not chunks or not isinstance(chunks, list):
        return None

    text = "\n".join([chunk["text"] for chunk in chunks if "text" in chunk]).strip()
    # replace consecutive newlines with a single newline
    text = re.sub(r"\n+", "\n", text)
    # replace consecutive spaces with a single space
    text = re.sub(r" +", " ", text)

    if not text:
        return None

    return text


async def generate_memory_prompt(
    client: AsyncOpenAI, instruction: str, row: Highlight, params: SamplingParams
) -> tuple[list[str], str] | None:
    source_meta = row["source_meta"]
    assert isinstance(source_meta, dict), "Source meta is not a dictionary"

    full_text = _get_full_text(row)

    if full_text is not None:
        logging.info(f"Using full text for row {row['task_id']} (len={len(full_text)})")
        user_prompt = _GENERATION_FULL_TEXT_REQUEST_TEMPLATE.format(
            title=source_meta["title"],
            url=row["source_url"],
            author=source_meta["author"],
            highlight=format_highlight_text(row),
            full_text=full_text,
        )
    else:
        logging.info(f"Using highlight interpretation for row {row['task_id']}")
        user_prompt = _GENERATION_INTERPRETATION_REQUEST_TEMPLATE.format(
            title=source_meta["title"],
            url=row["source_url"],
            author=source_meta["author"],
            highlight=format_highlight_text(row),
            highlight_interpretation=row["highlight_interpretation"],
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

    memory_prompts = parse_memory_prompts(result_text)
    if not memory_prompts:
        return [], result_text
    assert all(isinstance(mp, str) for mp in memory_prompts), "Memory prompts are not strings"

    return memory_prompts, result_text


class ArenaGeneratedMemoryPrompt(TypedDict):
    content: str
    judged_tier: MemoryPromptTier
    judged_reasoning: str


class _GenerationResult(Highlight):
    model: str
    model_prompts: list[ArenaGeneratedMemoryPrompt]
    model_completion: str


@dataclass
class _InferenceConfig:
    client: AsyncOpenAI
    params: SamplingParams


async def main(
    dataset: Dataset,
    generation: _InferenceConfig,
    grading: _InferenceConfig,
    instruction: str,
    instruction_name: str,
    out_dir: str,
    max_concurrency: int = 10,
):
    # Sanitize names for filename (replace / with _)
    safe_model_name = generation.params.model.replace("/", "_")
    safe_instruction_name = instruction_name.replace("/", "_")
    out_path = os.path.join(out_dir, f"generation_results_{safe_instruction_name}_{safe_model_name}.jsonl")

    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(out_path):
        keys = list(_GenerationResult.__annotations__.keys())
        df_existing = pd.DataFrame(columns=keys)  # pyright: ignore[reportArgumentType]
    else:
        df_existing = pd.read_json(out_path, lines=True)

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _process(row: Highlight) -> _GenerationResult | None:
        async with semaphore:
            try:
                result = await generate_memory_prompt(
                    client=generation.client,
                    instruction=instruction,
                    row=row,
                    params=generation.params,
                )
                if result is None:
                    logging.warning(f"No memory prompt generated for row {row['task_id']}")
                    return None

                generated_prompts, raw_generation_text = result

                judged_results = await asyncio.gather(
                    *[
                        classify_tier(
                            client=grading.client,
                            highlight=row,
                            references=row["references"],
                            target=generated_prompt,
                            params=grading.params,
                        )
                        for generated_prompt in generated_prompts
                    ]
                )
            except asyncio.TimeoutError:
                logging.warning(f"Timeout processing row {row['task_id']}")
                return None
            except Exception as e:
                logging.error(f"Error generating memory prompt for row {row['task_id']}: {e}")
                return None

        model_prompts: list[ArenaGeneratedMemoryPrompt] = [
            {
                "content": generated_prompt,
                "judged_tier": judged_result[0],
                "judged_reasoning": judged_result[1],
            }
            for generated_prompt, judged_result in zip(generated_prompts, judged_results)
        ]

        return {
            **row,
            "model": f"{generation.params.model} ({instruction_name})",
            "model_completion": raw_generation_text,
            "model_prompts": model_prompts,
        }

    # filter out tasks that already have results
    existing_task_ids = set(df_existing["task_id"].unique()) if len(df_existing) > 0 else set()
    dataset = dataset.filter(
        lambda x: x["task_id"] not in existing_task_ids and _get_full_text(x) is not None,
        desc="Filtering out tasks that already have results or have no full text",
    )
    print(f"After filtering: {len(dataset)} tasks remaining")

    generation_results = await tqdm.gather(
        *[_process(cast(Highlight, row)) for row in dataset],
        desc="Generating memory prompts",
    )

    # Filter out None results (from timeouts, errors, or failed card selection)
    valid_results = [r for r in generation_results if r is not None]
    logging.info(f"Successfully processed {len(valid_results)} out of {len(generation_results)} tasks")

    df = pd.concat([df_existing, pd.DataFrame(valid_results)])
    df.to_json(
        out_path,
        orient="records",
        lines=True,
    )


if __name__ == "__main__":
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="Generate and grade memory prompts for the arena.")

    # Required arguments
    parser.add_argument("--model", type=str, required=True, help="Generation model name")
    parser.add_argument(
        "--instruction-file",
        type=str,
        required=True,
        help="Path to instruction file for generation",
    )

    # Optional generation config
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (omitted if not specified)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="Reasoning effort for generation model",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom base URL for generation API",
    )
    parser.add_argument(
        "--auth-token",
        type=str,
        default=None,
        help="Custom authentication token for generation",
    )
    parser.add_argument(
        "--extra-body",
        type=str,
        default=None,
        help="Extra parameters as JSON string",
    )

    # Pipeline config
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Maximum concurrent operations",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="laddermedia/srs-highlights",
        help="HuggingFace dataset path",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (defaults to memory_machines/arena/results)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    load_dotenv()

    # Build generation client kwargs
    gen_client_kwargs = {}
    if args.base_url is not None:
        gen_client_kwargs["base_url"] = args.base_url
    if args.auth_token is not None:
        gen_client_kwargs["api_key"] = args.auth_token
    else:
        # Only assert OPENAI_API_KEY if no custom auth provided
        assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"

    generation_client = AsyncOpenAI(**gen_client_kwargs)

    # Grading client - ALWAYS uses Anthropic Claude Sonnet 4.5
    assert os.getenv("ANTHROPIC_API_KEY") is not None, (
        "ANTHROPIC_API_KEY environment variable must be set for grading (always uses Claude Sonnet 4.5)"
    )
    grading_client = AsyncOpenAI(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        base_url="https://api.anthropic.com/v1/",
    )

    # Load dataset
    dataset = load_dataset(args.dataset, split="grounded")
    assert isinstance(dataset, Dataset), "Dataset is not a Dataset"

    # Load instruction file for generation
    instruction_path = Path(args.instruction_file)
    if not instruction_path.is_absolute():
        # If relative path provided, resolve from repo root
        repo_root = Path(__file__).resolve().parents[2]
        instruction_path = repo_root / instruction_path

    if not instruction_path.exists():
        raise FileNotFoundError(
            f"Instruction file not found: {instruction_path}\n"
            f"Provide an absolute path or relative path from repo root."
        )

    with open(instruction_path, "r") as f:
        instruction = f.read()
    assert instruction.strip() != "", "Instruction file is empty"

    # Determine output directory
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        # Default to memory_machines/arena/results
        repo_root = Path(__file__).resolve().parents[2]
        out_dir = str(repo_root / "memory_machines" / "arena" / "results")

    # Parse generation params
    temperature = args.temperature if args.temperature is not None else Omit()

    # Parse extra_body if provided
    extra_kwargs = {}
    if args.extra_body is not None:
        try:
            extra_kwargs["extra_body"] = json.loads(args.extra_body)
            logging.info(f"Parsed extra_body for generation: {extra_kwargs}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse --extra-body as JSON: {e}")

    generation_params = SamplingParams(
        model=args.model,
        temperature=temperature,
        reasoning_effort=(args.reasoning_effort if args.reasoning_effort is not None else Omit()),
        extra_kwargs=extra_kwargs,
    )

    # Grading params - Always Claude Sonnet 4.5 with thinking
    grading_params = SamplingParams(
        model="claude-sonnet-4-5",
        temperature=Omit(),
        reasoning_effort="medium",  # Not used by Anthropic but kept for consistency
        extra_kwargs={"extra_body": {"thinking": {"type": "enabled", "budget_tokens": 2048}}},
    )

    # Extract instruction name from path (filename without extension)
    instruction_name = instruction_path.stem

    asyncio.run(
        main(
            dataset=dataset,
            generation=_InferenceConfig(
                client=generation_client,
                params=generation_params,
            ),
            grading=_InferenceConfig(
                client=grading_client,
                params=grading_params,
            ),
            instruction=instruction,
            instruction_name=instruction_name,
            out_dir=out_dir,
            max_concurrency=args.max_concurrency,
        )
    )
