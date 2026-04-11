"""
Trains reward models for pluckability preference learning on the srs-prompts dataset.

Uses TRL RewardTrainer with preference pairs derived from tier hierarchy (T3 > T2 > T1 > T0).

Example usage:
    python -m memory_machines.training.reward_model.train --base-model Qwen/Qwen2.5-0.5B --learning-rate 2e-5 --batch-size 4 --epochs 3

Run W&B sweep:
    wandb sweep memory_machines/training/reward_model/sweep.yaml
    wandb agent <sweep-id>
"""

import argparse
import logging

import torch
import wandb
from datasets import load_from_disk
from transformers import AutoTokenizer
from trl import RewardTrainer, RewardConfig


def evaluate_by_matchup_type(
    model,
    tokenizer,
    dataset,
    max_length: int,
    batch_size: int = 8,
    device: str = "cuda",
):
    """
    Evaluate reward model accuracy broken down by tier matchup type.

    For each preference pair, computes reward scores for chosen and rejected texts.
    Accuracy per matchup = percentage where reward(chosen) > reward(rejected).

    Args:
        model: Trained reward model (AutoModelForSequenceClassification with num_labels=1)
        tokenizer: Tokenizer used during training
        dataset: Test dataset split with chosen_tier and rejected_tier columns
        max_length: Maximum sequence length for tokenization
        batch_size: Batch size for inference
        device: Device to run inference on ("cuda" or "cpu")

    Returns:
        Tuple of (metrics_dict, predictions_list):
        - metrics_dict: Dictionary with eval_matchup/{type}_accuracy and _n metrics
        - predictions_list: List of dicts with per-example predictions for detailed logging
    """
    from collections import defaultdict

    logging.info(f"Running matchup-specific evaluation on {len(dataset)} pairs...")

    model.eval()
    model.to(device)

    # Store predictions and metadata
    all_predictions = []
    matchup_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    # Process in batches for efficiency
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]

        # Tokenize chosen texts
        chosen_encodings = tokenizer(
            batch["text_chosen"],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )

        # Tokenize rejected texts
        rejected_encodings = tokenizer(
            batch["text_rejected"],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )

        # Get rewards for chosen and rejected texts
        with torch.no_grad():
            chosen_inputs = {k: v.to(device) for k, v in chosen_encodings.items()}
            chosen_outputs = model(**chosen_inputs)
            chosen_rewards = chosen_outputs.logits.squeeze().tolist()
            # Handle single-element batch case
            if isinstance(chosen_rewards, float):
                chosen_rewards = [chosen_rewards]

            rejected_inputs = {k: v.to(device) for k, v in rejected_encodings.items()}
            rejected_outputs = model(**rejected_inputs)
            rejected_rewards = rejected_outputs.logits.squeeze().tolist()
            # Handle single-element batch case
            if isinstance(rejected_rewards, float):
                rejected_rewards = [rejected_rewards]

        # Process each example in batch
        for j in range(len(batch["chosen_tier"])):
            chosen_tier = batch["chosen_tier"][j]
            rejected_tier = batch["rejected_tier"][j]
            matchup_type = f"{chosen_tier}v{rejected_tier}"

            chosen_reward = chosen_rewards[j]
            rejected_reward = rejected_rewards[j]
            is_correct = chosen_reward > rejected_reward

            # Update matchup statistics
            matchup_stats[matchup_type]["correct"] += int(is_correct)
            matchup_stats[matchup_type]["total"] += 1

            # Store detailed prediction
            all_predictions.append(
                {
                    "index": i + j,
                    "chosen_tier": chosen_tier,
                    "rejected_tier": rejected_tier,
                    "matchup_type": matchup_type,
                    "chosen_reward": chosen_reward,
                    "rejected_reward": rejected_reward,
                    "correct": is_correct,
                    "source_url": batch["source_url"][j],
                    "highlight_id": batch["highlight_id"][j],
                }
            )

    # Calculate final metrics
    metrics = {}
    for matchup_type, stats in matchup_stats.items():
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"]
            metrics[f"eval_matchup/{matchup_type}_accuracy"] = accuracy
            metrics[f"eval_matchup/{matchup_type}_n"] = stats["total"]

    logging.info(f"Evaluated {len(all_predictions)} pairs across {len(matchup_stats)} matchup types")

    return metrics, all_predictions


def train_reward_model(
    data_dir: str,
    base_model: str,
    learning_rate: float = 2e-5,
    batch_size: int = 4,
    epochs: int = 3,
    max_length: int = 1024,
    seed: int = 42,
    wandb_project: str = "memory-machines-reward-model",
    run_name: str | None = None,
    resume_from_checkpoint: str | None = None,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
):
    """
    Train a reward model for pluckability preference learning.

    Args:
        data_dir: Path to prepared preference dataset
        base_model: HuggingFace model ID to use as base
        learning_rate: Learning rate
        batch_size: Per-device batch size
        epochs: Number of training epochs
        max_length: Maximum sequence length for tokenization
        seed: Random seed
        wandb_project: W&B project name
        run_name: W&B run name
        resume_from_checkpoint: Path to checkpoint to resume from
        push_to_hub: Whether to push model to HuggingFace Hub
        hub_model_id: HuggingFace Hub repository ID
    """
    # Initialize W&B
    wandb.init(
        project=wandb_project,
        name=run_name,
        config={
            "base_model": base_model,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "max_length": max_length,
            "seed": seed,
        },
    )

    # Load dataset
    logging.info(f"Loading dataset from: {data_dir}")
    dataset = load_from_disk(data_dir)
    logging.info(f"Dataset loaded: {dataset}")

    # Load tokenizer
    logging.info(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply chat template to preference pairs
    def apply_chat_template(examples):
        """Apply tokenizer chat template to chosen/rejected messages."""
        chosen_texts = [
            tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in examples["chosen"]
        ]
        rejected_texts = [
            tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in examples["rejected"]
        ]
        return {"text_chosen": chosen_texts, "text_rejected": rejected_texts}

    # Apply templates
    logging.info("Applying chat templates to preference pairs...")
    dataset = dataset.map(apply_chat_template, batched=True)

    # Verify tier columns are preserved
    logging.info(f"Dataset columns after applying chat template: {dataset['train'].column_names}")
    if "chosen_tier" not in dataset["test"].column_names:
        logging.warning("chosen_tier column not found in test set - matchup evaluation will fail!")

    assert wandb.run is not None, "Wandb run is not initialized"

    # Training arguments
    training_args = RewardConfig(
        output_dir=f"memory_machines/training/checkpoints/reward_model/{wandb.run.id}",
        max_length=max_length,
        model_init_kwargs={"dtype": torch.bfloat16},
        # Sweepable hyperparameters
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        # Fixed per essay specs
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,  # 5% warmup
        eval_strategy="steps",
        eval_steps=4,
        logging_steps=4,
        save_strategy="no",
        # Best model selection
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_accuracy",
        # greater_is_better=True,
        # save_steps=4,
        # save_total_limit=1,
        # W&B logging
        report_to="wandb",
        # Performance
        bf16=torch.cuda.is_available(),
        seed=seed,
        # Preserve metadata columns for post-training evaluation
        remove_unused_columns=False,
        # Hub
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
    )

    # Load model
    logging.info(f"Loading model: {base_model}")

    # Create RewardTrainer
    trainer = RewardTrainer(
        model=base_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )

    # Train
    logging.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Evaluate
    logging.info("Evaluating on test set...")
    metrics = trainer.evaluate()
    logging.info(f"Test metrics: {metrics}")

    # Matchup-specific evaluation
    logging.info("Computing matchup-specific accuracies...")
    matchup_metrics, predictions = evaluate_by_matchup_type(
        model=trainer.model,
        tokenizer=tokenizer,
        dataset=dataset["test"],
        max_length=max_length,
        batch_size=batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Log matchup metrics to W&B
    wandb.log(matchup_metrics)
    logging.info(f"Matchup-specific metrics: {matchup_metrics}")

    # Create summary table for W&B
    matchup_types = ["T3vT2", "T3vT1", "T2vT1", "T3vT0", "T2vT0", "T1vT0"]
    summary_data = []
    for matchup_type in matchup_types:
        accuracy_key = f"eval_matchup/{matchup_type}_accuracy"
        n_key = f"eval_matchup/{matchup_type}_n"
        if accuracy_key in matchup_metrics:
            summary_data.append(
                [
                    matchup_type,
                    matchup_metrics[accuracy_key],
                    matchup_metrics[n_key],
                ]
            )

    if summary_data:
        summary_table = wandb.Table(
            columns=["Matchup Type", "Accuracy", "Sample Count"],
            data=summary_data,
        )
        wandb.log({"eval_matchup_summary": summary_table})

    # Create detailed predictions table for W&B
    if predictions:
        detailed_data = [
            [
                pred["index"],
                pred["chosen_tier"],
                pred["rejected_tier"],
                pred["matchup_type"],
                pred["chosen_reward"],
                pred["rejected_reward"],
                pred["correct"],
                pred["source_url"],
                pred["highlight_id"],
            ]
            for pred in predictions
        ]
        detailed_table = wandb.Table(
            columns=[
                "Index",
                "Chosen Tier",
                "Rejected Tier",
                "Matchup Type",
                "Chosen Reward",
                "Rejected Reward",
                "Correct",
                "Source URL",
                "Highlight ID",
            ],
            data=detailed_data,
        )
        wandb.log({"eval_predictions_detailed": detailed_table})
        logging.info(f"Logged {len(predictions)} detailed predictions to W&B")

    # Save final model
    if push_to_hub:
        logging.info(f"Pushing model to Hub: {hub_model_id}")
        trainer.push_to_hub()

    wandb.finish()
    logging.info("Training complete!")

    return trainer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Train reward model for pluckability preference learning")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="memory_machines/training/data/reward_model",
        help="Path to prepared preference dataset",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., Qwen/Qwen2.5-0.5B)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="memory-machines-reward-model",
        help="W&B project name",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="W&B run name",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="HuggingFace Hub repository ID",
    )

    args = parser.parse_args()

    train_reward_model(
        data_dir=args.data_dir,
        base_model=args.base_model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_length=args.max_length,
        seed=args.seed,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        resume_from_checkpoint=args.resume_from_checkpoint,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )
