"""
Trains SFT models for memory prompt tier classification on the srs-highlights dataset.

4-class classification: T0 (off-target), T1 (needs-refactor), T2 (needs-polish), T3 (ready-to-review).
Uses LoRA fine-tuning with TRL SFTTrainer and assistant-only loss.

Example usage:
    python -m memory_machines.training.sft_tiering.train --lora_r 32 --learning_rate 2e-5 --batch_size 8

Run W&B sweep:
    wandb sweep memory_machines/training/sft_tiering/sweep.yaml
    wandb agent <sweep-id>
"""

import argparse
import logging
import re
import uuid

import torch
import wandb
from accelerate import PartialState
from datasets import load_from_disk
from peft import LoraConfig
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer


def run_inference_and_log(
    model,
    tokenizer,
    dataset,
    split_name: str,
    device: str = "cuda",
):
    """
    Run inference on a dataset split and log predictions vs ground truth to W&B.

    For each example:
    - Generate model prediction (T0/T1/T2/T3)
    - Compare to ground truth tier
    - Log detailed table with id, ground_truth, prediction, correct
    - Calculate per-tier accuracy and confusion matrix

    Args:
        model: Trained model (with LoRA adapters)
        tokenizer: Tokenizer used during training
        dataset: Dataset split with messages and tier columns
        split_name: Name of the split (e.g., "train", "test")
        batch_size: Not used (kept for compatibility), processes one at a time
        device: Device to run inference on ("cuda" or "cpu")
    """
    logging.info(f"Running inference on {split_name} split ({len(dataset)} examples)...")

    model.eval()
    model.to(device)

    all_predictions = []
    tier_stats = {tier: {"correct": 0, "total": 0} for tier in ["T0", "T1", "T2", "T3"]}

    for example in tqdm(dataset, desc=f"Inference on {split_name}"):
        # Format input (system + user messages only, exclude assistant message)
        messages = example["messages"][:-1]  # Exclude assistant message
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0,  # Greedy decoding for consistency
                do_sample=False,
            )

        # Decode prediction
        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        predicted_tier = generated_text.strip().upper()

        # Validate prediction - extract T0-T3 if present
        if predicted_tier not in ["T0", "T1", "T2", "T3"]:
            # Try to extract T0-T3 from the response
            match = re.search(r"T[0-3]", predicted_tier)
            predicted_tier = match.group(0) if match else "INVALID"

        ground_truth = example["tier"]
        is_correct = predicted_tier == ground_truth

        # Update statistics
        if ground_truth in tier_stats:
            tier_stats[ground_truth]["total"] += 1
            if is_correct:
                tier_stats[ground_truth]["correct"] += 1

        all_predictions.append(
            {
                "id": example["id"],
                "ground_truth": ground_truth,
                "predicted": predicted_tier,
                "correct": is_correct,
                "source_url": example.get("source_url", ""),
            }
        )

    # Calculate metrics
    total_correct = sum(p["correct"] for p in all_predictions)
    overall_accuracy = total_correct / len(all_predictions) if len(all_predictions) > 0 else 0.0

    # Per-tier accuracy
    tier_accuracies = {}
    for tier, stats in tier_stats.items():
        if stats["total"] > 0:
            tier_accuracies[tier] = stats["correct"] / stats["total"]
        else:
            tier_accuracies[tier] = 0.0

    # Log metrics to W&B
    metrics = {
        f"inference/{split_name}/overall_accuracy": overall_accuracy,
    }
    for tier, acc in tier_accuracies.items():
        metrics[f"inference/{split_name}/{tier}_accuracy"] = acc
        metrics[f"inference/{split_name}/{tier}_count"] = tier_stats[tier]["total"]

    wandb.log(metrics)

    # Create confusion matrix
    tier_labels = ["T0", "T1", "T2", "T3"]
    y_true = [p["ground_truth"] for p in all_predictions]
    # Replace INVALID with T0 for confusion matrix calculation
    y_pred = [p["predicted"] if p["predicted"] in tier_labels else "T0" for p in all_predictions]

    cm = confusion_matrix(y_true, y_pred, labels=tier_labels)

    # Log confusion matrix as W&B table
    cm_data = []
    for i, true_tier in enumerate(tier_labels):
        for j, pred_tier in enumerate(tier_labels):
            cm_data.append([true_tier, pred_tier, int(cm[i, j])])

    cm_table = wandb.Table(
        columns=["True Tier", "Predicted Tier", "Count"],
        data=cm_data,
    )
    wandb.log({f"inference/{split_name}/confusion_matrix": cm_table})

    # Log detailed predictions table
    pred_table = wandb.Table(
        columns=["id", "ground_truth", "predicted", "correct", "source_url"],
        data=[
            [p["id"], p["ground_truth"], p["predicted"], p["correct"], p["source_url"]] for p in all_predictions
        ],
    )
    wandb.log({f"predictions/{split_name}": pred_table})

    logging.info(f"Inference complete for {split_name}")
    logging.info(f"  Overall accuracy: {overall_accuracy:.3f}")
    for tier, acc in tier_accuracies.items():
        logging.info(f"  {tier} accuracy: {acc:.3f} (n={tier_stats[tier]['total']})")


def train_sft_tiering_model(
    data_dir: str,
    model_name: str,
    lora_r: int,
    learning_rate: float = 2e-5,
    batch_size: int = 8,
    epochs: int = 2,
    max_length: int = 2048,
    seed: int = 42,
    wandb_project: str = "memory-machines-sft-tiering",
    run_name: str | None = None,
    resume_from_checkpoint: str | None = None,
    inference_splits: list[str] | None = None,
):
    """
    Train a SFT model for tier classification with LoRA.

    Args:
        data_dir: Path to prepared dataset
        model_name: HuggingFace model ID to use (e.g., OpenPipe/Qwen3-14B-Instruct)
        lora_r: LoRA rank (8, 32, or 64)
        learning_rate: Learning rate
        batch_size: Per-device batch size
        epochs: Number of training epochs
        max_length: Maximum sequence length for tokenization
        seed: Random seed
        wandb_project: W&B project name
        run_name: W&B run name
        resume_from_checkpoint: Path to checkpoint to resume from
        inference_splits: List of splits to run inference on (default: ["train", "test"])
    """
    # Load dataset - resolve path relative to project root if needed
    from pathlib import Path

    data_path = Path(data_dir)
    if not data_path.exists():
        # Try resolving from project root
        project_root = Path(__file__).parent.parent.parent.parent
        data_path = project_root / data_dir

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found at {data_dir} or {data_path}. "
            f"Please run: python -m memory_machines.training.sft_tiering.prepare_dataset"
        )

    logging.info(f"Loading dataset from: {data_path}")
    dataset = load_from_disk(str(data_path))
    logging.info(f"Dataset loaded: {dataset}")

    # Load tokenizer
    logging.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load custom chat template (hard-coded for Qwen3)
    chat_template_path = Path(__file__).parent / "qwen3-instruct.jinja2"
    logging.info(f"Loading chat template from: {chat_template_path}")
    with open(chat_template_path, "r") as f:
        tokenizer.chat_template = f.read()

    # Create output_dir before wandb init (all processes need the same path)
    run_id = run_name if run_name else str(uuid.uuid4())[:6]
    output_dir = f"memory_machines/training/checkpoints/sft_tiering/{run_id}"

    lora_alpha = 32

    # Initialize W&B (only on main process for distributed training)
    if PartialState().is_main_process:
        wandb.init(
            project=wandb_project,
            name=run_id,
            config={
                "model_name": model_name,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "max_length": max_length,
                "seed": seed,
            },
        )
        assert wandb.run is not None, "wandb.run is not initialized"

    # Configure LoRA (MANDATORY - always use LoRA)
    logging.info(f"Configuring LoRA with rank={lora_r}, alpha={lora_alpha}")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
    )

    # Training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        packing=False,  # Don't pack multiple examples together to avoid truncation errors
        # Sweepable hyperparameters
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,  # 5% warmup
        eval_strategy="steps",
        eval_steps=10,
        logging_steps=10,
        save_strategy="steps",
        save_steps=20,
        save_total_limit=1,
        # Best model selection
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # W&B logging
        report_to=["wandb"] if PartialState().is_main_process else [],
        # Performance
        bf16=True,
        fp16=False,
        seed=seed,
        # Model initialization
        model_init_kwargs={"dtype": torch.bfloat16},
        # Preserve metadata columns for post-training evaluation
        remove_unused_columns=False,
        # Assistant-only loss (compute loss only on assistant response)
        # NOTE: datasets need to be in the Chat format for this to work and for
        # tokenizer chat template to include the assistant response mask internally
        assistant_only_loss=True,
    )

    # Create SFTTrainer (pass model_name as string, not loaded model)
    trainer = SFTTrainer(
        model=model_name,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # Print number of trainable parameters
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    # Train
    logging.info("Starting training...")
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except Exception as e:
        if PartialState().is_main_process:
            wandb.finish(exit_code=1)
        raise e

    # Evaluate
    logging.info("Evaluating on test set...")
    metrics = trainer.evaluate()
    logging.info(f"Test metrics: {metrics}")

    # Save final model (only on main process)
    final_model_path = f"{output_dir}/final"
    if PartialState().is_main_process:
        logging.info(f"Saving final model to: {final_model_path}")
        trainer.model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        logging.info(f"✅ Model saved to {final_model_path}")

    # Run inference on train and test splits
    if inference_splits is None:
        inference_splits = ["train", "test"]

    for split in inference_splits:
        if split not in dataset:
            logging.warning(f"Split '{split}' not found in dataset, skipping...")
            continue
        run_inference_and_log(
            model=trainer.model,
            tokenizer=tokenizer,
            dataset=dataset[split],
            split_name=split,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    if PartialState().is_main_process:
        logging.info("Uploading adapter as W&B artifact...")
        assert wandb.run is not None, "wandb.run is not initialized"

        from pathlib import Path

        adapter_path = Path(output_dir) / "adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)

        logging.info(f"Saving adapter to {adapter_path}...")
        trainer.model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))

        artifact = wandb.Artifact(
            name=f"sft-tiering-adapter-lora-{lora_config.r}",
            type="lora",
            description="LoRA adapter trained on the tiering dataset via SFT",
            metadata={
                "wandb.base_model": model_name,
                "dataset": "laddermedia/srs-highlights",
                "lora_r": lora_config.r,
                "lora_alpha": lora_config.lora_alpha,
                "global_step": trainer.state.global_step,
                "resumed_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint else None,
            },
            storage_region="coreweave-us",
        )
        artifact.add_dir(str(adapter_path))
        wandb.log_artifact(artifact)
        logging.info("Adapter artifact logged successfully!")

        artifact.wait()
        logging.info("Adapter artifact upload complete!")
        wandb.finish(exit_code=0)

    logging.info("Training complete!")

    return trainer


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Train SFT model for tier classification")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="memory_machines/training/data/sft_tiering",
        help="Path to prepared dataset",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="OpenPipe/Qwen3-14B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        required=True,
        choices=[1, 8, 32, 64, 128],
        help="LoRA rank (1, 8, 32, 64 or 128)",
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
        default=8,
        help="Per-device batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
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
        default="memory-machines-sft-tiering",
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
        "--inference_splits",
        type=str,
        nargs="+",
        default=["train", "test"],
        help="List of splits to run inference on (e.g., train test)",
    )

    args = parser.parse_args()

    train_sft_tiering_model(
        data_dir=args.data_dir,
        model_name=args.model_name,
        lora_r=args.lora_r,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_length=args.max_length,
        seed=args.seed,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        resume_from_checkpoint=args.resume_from_checkpoint,
        inference_splits=args.inference_splits,
    )
