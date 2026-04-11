"""
Evaluation utilities for comparing instruction prompts on tier classification.

Can be used to compare hand-crafted vs GEPA-optimized instructions.

Example usage:
    python -m memory_machines.training.optimized_tiering.evaluate \
        --data-dir memory_machines/training/data/optimized_tiering \
        --optimized-program memory_machines/training/checkpoints/optimized_tiering/{run_id}/optimized_program.json
"""

import argparse
import logging
import os
from pathlib import Path

import dspy

from memory_machines.training.optimized_tiering.prepare_dataset import load_prepared_dataset
from memory_machines.training.optimized_tiering.train import ClassifyMemoryPrompt, metric


def evaluate_program(
    program_path: str,
    data_dir: str,
    api_key: str = None,
    use_val_set: bool = True,
):
    """
    Evaluate a saved DSPy program on the test/val set.

    Args:
        program_path: Path to saved program JSON
        data_dir: Path to prepared dataset
        api_key: OpenAI API key
        use_val_set: If True, evaluate on val set; otherwise on train set
    """
    # Configure DSPy LM
    lm = dspy.LM("openai/gpt-4o-mini", temperature=0.0, max_tokens=4000, api_key=api_key)
    dspy.configure(lm=lm)

    # Load dataset
    logging.info("Loading dataset...")
    train_set, val_set = load_prepared_dataset(data_dir)
    eval_set = val_set if use_val_set else train_set
    set_name = "validation" if use_val_set else "train"
    logging.info(f"Evaluating on {set_name} set: {len(eval_set)} examples")

    # Load program
    logging.info(f"Loading program from: {program_path}")
    program = dspy.ChainOfThought(ClassifyMemoryPrompt)
    program.load(program_path)

    # Evaluate
    logging.info("Evaluating program...")
    evaluator = dspy.Evaluate(
        devset=eval_set,
        metric=metric,
        num_threads=8,
        display_progress=True,
        display_table=True,
    )
    score = evaluator(program)

    logging.info(f"\n{'='*80}")
    logging.info(f"Evaluation Results ({set_name} set)")
    logging.info(f"{'='*80}")
    logging.info(f"Accuracy: {score:.4f}")
    logging.info(f"Total examples: {len(eval_set)}")
    logging.info(f"{'='*80}")

    return score


def compare_instructions(
    baseline_program_path: str,
    optimized_program_path: str,
    data_dir: str,
    api_key: str = None,
):
    """
    Compare baseline and optimized programs side by side.

    Args:
        baseline_program_path: Path to baseline program JSON
        optimized_program_path: Path to optimized program JSON
        data_dir: Path to prepared dataset
        api_key: OpenAI API key
    """
    logging.info("Comparing baseline vs optimized programs...")

    # Evaluate baseline
    logging.info("\n" + "=" * 80)
    logging.info("BASELINE PROGRAM")
    logging.info("=" * 80)
    baseline_score = evaluate_program(baseline_program_path, data_dir, api_key)

    # Evaluate optimized
    logging.info("\n" + "=" * 80)
    logging.info("OPTIMIZED PROGRAM")
    logging.info("=" * 80)
    optimized_score = evaluate_program(optimized_program_path, data_dir, api_key)

    # Summary
    improvement = optimized_score - baseline_score
    improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0

    logging.info("\n" + "=" * 80)
    logging.info("COMPARISON SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Baseline:  {baseline_score:.4f}")
    logging.info(f"Optimized: {optimized_score:.4f}")
    logging.info(f"Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")
    logging.info("=" * 80)

    return baseline_score, optimized_score


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate DSPy programs for tier classification")
    parser.add_argument("--data-dir", required=True, help="Path to prepared dataset")
    parser.add_argument(
        "--optimized-program",
        type=str,
        required=True,
        help="Path to optimized program JSON",
    )
    parser.add_argument(
        "--baseline-program",
        type=str,
        default=None,
        help="Path to baseline program JSON (for comparison)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (or use OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--use-train-set",
        action="store_true",
        help="Evaluate on train set instead of val set",
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key must be provided via --api-key or OPENAI_API_KEY env var")

    # Run evaluation
    if args.baseline_program:
        compare_instructions(args.baseline_program, args.optimized_program, args.data_dir, api_key)
    else:
        evaluate_program(
            args.optimized_program,
            args.data_dir,
            api_key,
            use_val_set=not args.use_train_set,
        )
