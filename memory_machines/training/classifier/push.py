"""
Push a trained pluckability classifier to HuggingFace Hub.

Example usage:
    python -m memory_machines.training.classifier.push \
        --checkpoint_path memory_machines/training/checkpoints/classifier/abc123/final \
        --hub_model_id your-org/pluckability-classifier
"""

import argparse
import logging

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def push_to_hub(
    checkpoint_path: str,
    hub_model_id: str,
    private: bool = False,
):
    """
    Load a trained model from checkpoint and push to HuggingFace Hub.

    Args:
        checkpoint_path: Path to the trained model checkpoint
        hub_model_id: HuggingFace Hub repository ID (e.g., 'org/model-name')
        private: Whether to create a private repository
    """
    logging.info(f"Loading model from: {checkpoint_path}")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path, dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    logging.info(f"Pushing model to Hub: {hub_model_id}")
    model.push_to_hub(hub_model_id, private=private)
    tokenizer.push_to_hub(hub_model_id, private=private)

    logging.info(f"Successfully pushed to: https://huggingface.co/{hub_model_id}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Push trained classifier to HuggingFace Hub")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        required=True,
        help="HuggingFace Hub repository ID (e.g., 'org/model-name')",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )

    args = parser.parse_args()

    push_to_hub(
        checkpoint_path=args.checkpoint_path,
        hub_model_id=args.hub_model_id,
        private=args.private,
    )
