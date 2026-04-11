"""
Produces an optimized instruction for grounded memory prompt tier classification.

Uses DSPy's GEPA (Generalized Prompt Evolution Algorithm) to automatically optimize
the instruction prompt for classifying memory prompts into tiers T0-T3.

Example usage:
    python -m memory_machines.training.optimized_tiering.train --data-dir memory_machines/training/data/optimized_tiering
"""

import argparse
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import dspy

from memory_machines.training.optimized_tiering.prepare_dataset import load_prepared_dataset


class ClassifyMemoryPrompt(dspy.Signature):
    """Classify a memory prompt for spaced repetition into tiers T0-T3 based on quality."""

    source_description = dspy.InputField(
        desc="Source text context including title, author, highlight, and interpretation"
    )
    reference_cards = dspy.InputField(desc="Past rated memory prompts from the same highlight for calibration")
    target_prompt = dspy.InputField(desc="The memory prompt to evaluate")
    tier = dspy.OutputField(
        desc="Classification tier (T0/T1/T2/T3) based on targeting accuracy and construction quality"
    )


def metric(example, prediction, trace=None):
    """Basic metric: check if predicted tier matches correct tier."""
    try:
        predicted_tier = prediction.tier.strip().upper()
        if predicted_tier not in ["T0", "T1", "T2", "T3"]:
            return 0
        return int(predicted_tier == example.correct_tier)
    except Exception:
        return 0


def metric_with_feedback(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Metric with feedback for GEPA optimization."""
    correct_tier = gold.correct_tier

    try:
        predicted_tier = pred.tier.strip().upper()
        if predicted_tier not in ["T0", "T1", "T2", "T3"]:
            feedback = f"Invalid tier '{pred.tier}'. Must output exactly T0, T1, T2, or T3. "
            feedback += f"The correct tier is {correct_tier}."
            return dspy.Prediction(score=0, feedback=feedback)

        score = int(predicted_tier == correct_tier)

        if score == 1:
            # Provide tier-specific reasoning for correct predictions
            feedback = f"Correct! The tier is {correct_tier}. "
            if correct_tier == "T3":
                feedback += "You correctly identified this prompt as well-targeted to user interest AND well-constructed for spaced repetition. The question is specific, unambiguous, and provides sufficient contextual cues."
            elif correct_tier == "T2":
                feedback += "You correctly identified this as semantically aligned with user interest but needing minor polish. The core targeting and construction are sound—only word-level improvements needed."
            elif correct_tier == "T1":
                feedback += "You correctly identified this as in the right region of meaning but ineffective for review. The targeting may be close, but the question construction fails reviewability tests (too vague, lacks context, or ambiguous)."
            elif correct_tier == "T0":
                feedback += "You correctly identified this as off-target. The prompt focuses on details not aligned with the user's likely interest—either macro misalignment (wrong aspect entirely) or micro misalignment (nearby but irrelevant details)."
        else:
            # Provide tier-specific guidance for incorrect predictions
            feedback = f"Incorrect. Predicted {predicted_tier}, but correct tier is {correct_tier}. "

            # Explain what the correct tier means
            if correct_tier == "T3":
                feedback += "This prompt is actually T3 because it satisfies BOTH criteria: (1) well-targeted to user interest, AND (2) well-constructed for review (specific question, unambiguous, sufficient cues). "
            elif correct_tier == "T2":
                feedback += "This prompt is actually T2 because it's semantically aligned with user interest and reviewable, but needs minor wording/phrasing improvements. Core structure is sound—only polish needed. "
            elif correct_tier == "T1":
                feedback += "This prompt is actually T1 because while it may be in the right region, it fails reviewability tests. The question is too vague/generic, lacks contextual cues, could elicit multiple answers, or loses source-specific anchors. "
            elif correct_tier == "T0":
                feedback += "This prompt is actually T0 because it's off-target—focusing on details the user doesn't care about, not what they highlighted. "

            # Add guidance on common confusion patterns
            if predicted_tier == "T3" and correct_tier == "T2":
                feedback += "Don't conflate 'semantically correct' with 'ready to review'. T2 means it would work but isn't optimal yet."
            elif predicted_tier == "T3" and correct_tier == "T1":
                feedback += "Focus on question quality: Is it specific enough? Does it have sufficient contextual cues? Can it elicit multiple valid answers? If any fail, it's T1."
            elif predicted_tier == "T2" and correct_tier == "T1":
                feedback += "T1 vs T2 distinction: T2 needs word-level changes only. T1 needs structural changes (scope, framing, knowledge components). If the question itself is fundamentally vague or generic, it's T1."
            elif predicted_tier == "T1" and correct_tier == "T0":
                feedback += "T0 vs T1 distinction: T1 is in the right region but poorly constructed. T0 targets the wrong information entirely—different aspect of the text than what user cared about."

            feedback += "\n\nRemember: Evaluate BOTH targeting (user interest alignment) AND construction (question specificity, contextual cues, reviewability)."

        return dspy.Prediction(score=score, feedback=feedback)
    except Exception as e:
        feedback = f"Error processing prediction: {str(e)}. Correct tier is {correct_tier}."
        return dspy.Prediction(score=0, feedback=feedback)


def train_dspy_optimizer(
    data_dir: str,
    lm_api_key: str,
    reflection_lm_api_key: str,
    wandb_project: str = "memory-machines-dspy",
    run_name: str | None = None,
    seed: int = 42,
):
    """Train GEPA optimizer for memory prompt classification."""

    config = {
        "model": "anthropic/claude-sonnet-4-5",  # Program LM
        "reflection_model": "openai/gpt-5.2",  # Reflection LM
        "auto": "light",
        "reflection_minibatch_size": 6,
        "num_threads": 32,
        "seed": seed,
        "wandb_project": wandb_project,
    }
    config["run_name"] = run_name or f"gepa-{config['auto']}-{config['model']}"

    # Set random seed
    random.seed(seed)

    # Configure DSPy LM for the program
    lm = dspy.LM(config["model"], api_key=lm_api_key)
    dspy.configure(lm=lm)

    # Load prepared dataset
    logging.info("Loading dataset...")
    train_set, val_set = load_prepared_dataset(data_dir)
    logging.info(f"Train: {len(train_set)}, Val: {len(val_set)}")

    # Create program
    logging.info("Creating program...")
    program = dspy.ChainOfThought(ClassifyMemoryPrompt)

    # Evaluate baseline
    logging.info("Evaluating baseline...")
    baseline_eval = dspy.Evaluate(
        devset=val_set, metric=metric, num_threads=config["num_threads"], display_progress=True
    )
    baseline_score = baseline_eval(program)
    logging.info(f"Baseline score: {baseline_score}")

    # Initialize GEPA optimizer with reflection LM
    logging.info("Initializing GEPA optimizer...")
    reflection_lm = dspy.LM(config["reflection_model"], api_key=reflection_lm_api_key)

    optimizer = dspy.GEPA(
        metric=metric_with_feedback,
        auto=config["auto"],
        num_threads=config["num_threads"],
        track_stats=True,
        reflection_minibatch_size=config["reflection_minibatch_size"],
        reflection_lm=reflection_lm,
        use_wandb=True,
        wandb_init_kwargs={
            "project": config["wandb_project"],
            "name": config["run_name"],
        },
    )

    # Compile (optimize) the program
    logging.info("Optimizing with GEPA...")
    optimized_program = optimizer.compile(program, trainset=train_set, valset=val_set)

    # Save compiled program immediately
    logging.info("Saving compiled program...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"memory_machines/training/checkpoints/optimized_tiering/{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    optimized_program.save(str(output_dir / "optimized_program.json"))

    # Save the optimized instruction
    optimized_instruction = optimized_program.predict.signature.instructions  # type: ignore
    with open(output_dir / "optimized_instruction.txt", "w") as f:
        f.write(optimized_instruction)

    # Evaluate optimized program
    logging.info("Evaluating optimized program...")
    optimized_eval = dspy.Evaluate(
        devset=val_set,
        metric=metric,
        num_threads=config["num_threads"],
        display_progress=True,
        display_table=True,
    )
    optimized_score = optimized_eval(optimized_program)
    logging.info(f"Optimized score: {optimized_score}")

    logging.info(f"Saved optimized program to {output_dir}")
    logging.info("\n" + "=" * 80)
    logging.info("OPTIMIZED INSTRUCTION:")
    logging.info("=" * 80)
    logging.info(optimized_instruction)
    logging.info("=" * 80)

    return optimized_program


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Train GEPA optimizer for memory prompt classification (hard-coded config)"
    )
    parser.add_argument("--data-dir", required=True, help="Path to prepared dataset")
    parser.add_argument("--wandb-project", default="memory-machines-dspy", help="W&B project name")
    parser.add_argument("--run-name", default=None, help="W&B run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lm-api-key", default=None, help="OpenAI API key (or use OPENAI_API_KEY env var)")
    parser.add_argument(
        "--reflection-lm-api-key", default=None, help="Reflection LM API key (or use OPENAI_API_KEY env var)"
    )

    args = parser.parse_args()

    # Use API key from args or environment
    lm_api_key = args.lm_api_key or os.getenv("OPENAI_API_KEY")
    if not lm_api_key:
        raise ValueError("OpenAI API key must be provided via --api-key or OPENAI_API_KEY env var")

    reflection_lm_api_key = args.reflection_lm_api_key or os.getenv("OPENAI_API_KEY")
    if not reflection_lm_api_key:
        raise ValueError(
            "Reflection LM API key must be provided via --reflection-lm-api-key or OPENAI_API_KEY env var"
        )

    train_dspy_optimizer(
        data_dir=args.data_dir,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        lm_api_key=lm_api_key,
        reflection_lm_api_key=reflection_lm_api_key,
        seed=args.seed,
    )
