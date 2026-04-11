"""
Trains classifiers for pluckability prediction on the srs-prompts dataset.

Binary classification: unpluckable (T0/T1) vs pluckable (T2/T3).
Supports both full fine-tuning (BERT-style models) and LoRA fine-tuning (larger LLMs).

Example usage:
    python -m memory_machines.training.classifier.train --model_name answerdotai/ModernBERT-base --learning_rate 2e-5 --batch_size 16 --epochs 3
    python -m memory_machines.training.classifier.train --model_name OpenPipe/Qwen3-14B-Instruct --lora_r 8 --learning_rate 5e-5 --batch_size 4 --epochs 2

Run W&B sweep:
    wandb sweep memory_machines/training/classifier/sweep.yaml
    wandb agent <sweep-id>
"""

import argparse
import logging

import numpy as np
import torch
import wandb
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from memory_machines.utils.format import format_user_message


def softmax(x, axis=-1):
    """Compute softmax values for array x along specified axis."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _validate_rendered_length(dataset, tokenizer, max_length: int, input_format: str):
    """
    Validate that tokenized examples don't exceed max_length.

    Logs average, maximum, and percentage of examples exceeding max_length.
    Warns if average length exceeds max_length or if >10% of examples exceed limit.
    """
    logging.info("Validating rendered text lengths...")

    def _calculate_rendered_len(row):
        # Format text according to input_format
        if input_format == "content-only":
            text = row["content"]
        else:  # full-context
            text = format_user_message(row)

        # Apply chat template if available (no-op for BERT-style models)
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=False,
            )

        # Tokenize without truncation to get actual length
        tokens = tokenizer(text, truncation=False)
        return {"rendered_len": len(tokens["input_ids"])}

    # Calculate lengths for all examples
    rendered_dataset = dataset.map(_calculate_rendered_len)
    rendered_lengths = rendered_dataset["rendered_len"]

    # Compute statistics
    avg_len = sum(rendered_lengths) / len(rendered_lengths)
    max_len = max(rendered_lengths)
    pct_exceeds = sum(1 for l in rendered_lengths if l > max_length) / len(rendered_lengths) * 100

    # Log statistics
    logging.info("Tokenized length statistics:")
    logging.info(f"  Average: {avg_len:.1f} tokens")
    logging.info(f"  Maximum: {max_len} tokens")
    logging.info(f"  % exceeding max_length ({max_length}): {pct_exceeds:.1f}%")

    # Warn if issues detected
    if avg_len > max_length:
        logging.warning(f"Average length ({avg_len:.1f}) exceeds max_length ({max_length})")
    if pct_exceeds > 10:
        logging.warning(
            f"Over 10% of examples exceed max_length ({pct_exceeds:.1f}%) - consider increasing max_length"
        )


def compute_metrics(eval_pred):
    """Compute classification metrics including ROC-AUC."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    probs = softmax(logits, axis=-1)[:, 1]  # Pluckable class probability

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs),  # Primary metric
    }

    # Log ROC curve to W&B
    fpr, tpr, _ = roc_curve(labels, probs)
    wandb.log(
        {
            "roc_curve": wandb.plot.line_series(
                xs=fpr.tolist(),
                ys=[tpr.tolist()],
                keys=["TPR"],
                title="ROC Curve",
                xname="FPR",
            )
        }
    )

    return metrics


def run_inference_and_log(
    model, dataset, split_name: str, data_collator, batch_size: int = 32, device: str = "cuda"
):
    """
    Run inference on a dataset split and log predictions table to W&B.

    Args:
        model: The model to use for inference
        dataset: Tokenized dataset split
        split_name: Name of the split (e.g., 'train', 'test')
        data_collator: Data collator for batching
        batch_size: Batch size for inference
        device: Device to run inference on
    """
    logging.info(f"Running inference on {split_name} split...")

    model.eval()
    model.to(device)

    # Create dataloader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    all_logits = []
    all_labels = []
    all_ids = []

    # Run inference
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            ids = batch["id"]

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()

            all_logits.append(logits)
            all_labels.append(labels.numpy())
            all_ids.extend(ids)

    # Concatenate all batches
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Calculate probabilities
    probs = softmax(all_logits, axis=-1)
    prob_unpluckable = probs[:, 0]
    prob_pluckable = probs[:, 1]
    predicted_labels = np.argmax(all_logits, axis=-1)

    # Create wandb table
    table_data = []
    for id_val, label, pred, prob_0, prob_1 in zip(all_ids, all_labels, predicted_labels, prob_unpluckable, prob_pluckable):
        table_data.append([id_val, int(label), int(pred), float(prob_0), float(prob_1)])

    table = wandb.Table(
        columns=["id", "true_label", "predicted_label", "prob_unpluckable", "prob_pluckable"],
        data=table_data,
    )

    # Log table to W&B
    wandb.log({f"predictions/{split_name}": table})
    logging.info(f"Logged {len(table_data)} predictions for {split_name} split to W&B")

    # Also log aggregate metrics for this split
    metrics = {
        "accuracy": accuracy_score(all_labels, predicted_labels),
        "precision": precision_score(all_labels, predicted_labels, zero_division=0),
        "recall": recall_score(all_labels, predicted_labels, zero_division=0),
        "f1": f1_score(all_labels, predicted_labels, zero_division=0),
        "roc_auc": roc_auc_score(all_labels, prob_pluckable),
    }
    wandb.log({f"inference_metrics/{split_name}": metrics})
    logging.info(f"Inference metrics for {split_name}: {metrics}")


def train_pluckability_classifier(
    data_dir: str,
    model_name: str,
    input_format: str = "full-context",
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    epochs: int = 3,
    max_length: int = 1024,
    seed: int = 42,
    lora_r: int | None = None,
    wandb_project: str = "memory-machines-pluckability-classifier",
    run_name: str | None = None,
    resume_from_checkpoint: str | None = None,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
    inference_checkpoint: str | None = None,
    inference_splits: list[str] | None = None,
):
    # Initialize W&B
    wandb.init(
        project=wandb_project,
        name=run_name,
        config={
            "model_name": model_name,
            "input_format": input_format,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "max_length": max_length,
            "seed": seed,
            "lora_r": lora_r,
        },
    )

    # Load dataset
    logging.info(f"Loading dataset from: {data_dir}")
    dataset = load_from_disk(data_dir)
    assert isinstance(dataset, DatasetDict), "Dataset is not a DatasetDict"
    logging.info(f"Dataset loaded: {dataset}")

    # Load tokenizer
    logging.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Causal LMs classify from the last token, so pad on the left
    if lora_r:
        tokenizer.padding_side = "left"

    # Validate rendered lengths on train split
    _validate_rendered_length(dataset["train"], tokenizer, max_length, input_format)

    # Tokenization function
    def tokenize_function(examples):
        if input_format == "content-only":
            texts = examples["content"]
        else:  # full-context
            texts = [
                format_user_message(row)
                for row in [dict(zip(examples.keys(), values)) for values in zip(*examples.values())]
            ]

        # Apply chat template if available (no-op for BERT-style models)
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for text in texts
            ]

        return tokenizer(texts, truncation=True, max_length=max_length)

    # Tokenize dataset
    logging.info("Tokenizing dataset...")
    columns_to_remove = [col for col in dataset["train"].column_names if col not in ["label", "id"]]
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=columns_to_remove)
    tokenized = tokenized.rename_column("label", "labels")

    # Load model
    logging.info(f"Loading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        torch_dtype=torch.bfloat16 if lora_r else None,
    )

    # Set pad token ID
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Apply LoRA if requested
    if lora_r:
        logging.info(f"Applying LoRA with rank={lora_r}, alpha=32")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            bias="none",
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"memory_machines/training/checkpoints/classifier/{wandb.run.id}",
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
        save_strategy="steps",
        save_steps=8,
        save_total_limit=1,
        # Best model selection
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        # W&B logging
        report_to="wandb",
        # Performance
        bf16=torch.cuda.is_available(),
        seed=seed,
        # Hub
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    logging.info("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Evaluate
    logging.info("Evaluating on test set...")
    metrics = trainer.evaluate()
    logging.info(f"Test metrics: {metrics}")

    # Save final model locally
    final_model_path = f"memory_machines/training/checkpoints/classifier/{wandb.run.id}/final"
    logging.info(f"Saving final model to: {final_model_path}")
    trainer.save_model(final_model_path)

    # Save final model to hub if requested
    if push_to_hub:
        logging.info(f"Pushing model to Hub: {hub_model_id}")
        trainer.push_to_hub()

    # Run inference on specified splits if requested
    if inference_splits:
        if inference_checkpoint:
            logging.info(f"Loading checkpoint from {inference_checkpoint} for inference...")
            inference_model = AutoModelForSequenceClassification.from_pretrained(inference_checkpoint)
            if inference_model.config.pad_token_id is None:
                inference_model.config.pad_token_id = tokenizer.pad_token_id
        else:
            logging.info("Using trained model for inference...")
            inference_model = trainer.model

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        for split in inference_splits:
            if split not in tokenized:
                logging.warning(f"Split '{split}' not found in dataset, skipping...")
                continue
            run_inference_and_log(
                inference_model, tokenized[split], split, data_collator, batch_size=batch_size, device=device
            )

    wandb.finish()
    logging.info("Training complete!")

    return trainer, final_model_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Train BERT-style pluckability classifier")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="memory_machines/training/data/classifier",
        help="Path to prepared dataset",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., answerdotai/ModernBERT-base)",
    )
    parser.add_argument(
        "--input_format",
        type=str,
        default="full-context",
        choices=["content-only", "full-context"],
        help="Input format: content-only (just memory prompt) or full-context (with highlight/source)",
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
        default=16,
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
        "--lora_r",
        type=int,
        default=None,
        help="LoRA rank. When provided, uses LoRA fine-tuning instead of full fine-tuning.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="memory-machines-pluckability-classifier",
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
    parser.add_argument(
        "--inference_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for post-training inference (e.g., best model checkpoint)",
    )
    parser.add_argument(
        "--inference_splits",
        type=str,
        nargs="+",
        default=["test"],
        help="Dataset splits to run inference on (e.g., train test)",
    )

    args = parser.parse_args()

    train_pluckability_classifier(
        data_dir=args.data_dir,
        model_name=args.model_name,
        input_format=args.input_format,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_length=args.max_length,
        seed=args.seed,
        lora_r=args.lora_r,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        resume_from_checkpoint=args.resume_from_checkpoint,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        inference_checkpoint=args.inference_checkpoint,
        inference_splits=args.inference_splits,
    )
