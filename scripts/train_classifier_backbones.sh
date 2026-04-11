#!/usr/bin/env bash
# Train classifier models with specified hyperparameters and run inference
#
# This script trains three backbone models:
# - Qwen/Qwen3-0.6B: 2e-5 LR, 8 BS, 3 epochs (target: 0.83 ROC-AUC)
# - answerdotai/ModernBERT-base: 2e-5 LR, 8 BS, 3 epochs (target: 0.72 ROC-AUC)
# - answerdotai/ModernBERT-large: 5e-6 LR, 8 BS, 2 epochs (target: 0.69 ROC-AUC)
#
# Usage: ./scripts/train_classifier_backbones.sh [--inference-splits train test]

set -e

# Parse arguments
INFERENCE_SPLITS="train test"
while [[ $# -gt 0 ]]; do
    case $1 in
        --inference-splits)
            shift
            INFERENCE_SPLITS="$@"
            break
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--inference-splits train test]"
            exit 1
            ;;
    esac
    shift
done

# Configuration
DATA_DIR="memory_machines/training/data/classifier"
INPUT_FORMAT="full-context"
MAX_LENGTH=2048
SEED=42
WANDB_PROJECT="memory-machines-pluckability-classifier"

echo "========================================="
echo "Training Classifier Backbones"
echo "========================================="
echo "Data Directory: $DATA_DIR"
echo "Input Format: $INPUT_FORMAT"
echo "Max Length: $MAX_LENGTH"
echo "Inference Splits: $INFERENCE_SPLITS"
echo "========================================="
echo ""

# Model 1: Qwen/Qwen3-0.6B
echo "========================================="
echo "Training Model 1/3: Qwen/Qwen3-0.6B"
echo "Hyperparameters: LR=2e-5, BS=8, Epochs=3"
echo "Target ROC-AUC: 0.83"
echo "========================================="
python -m memory_machines.training.classifier.train \
    --data_dir "$DATA_DIR" \
    --model_name "Qwen/Qwen3-0.6B" \
    --input_format "$INPUT_FORMAT" \
    --learning_rate 2e-5 \
    --batch_size 8 \
    --epochs 3 \
    --max_length "$MAX_LENGTH" \
    --seed "$SEED" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "qwen3-0.6b-final" \
    --inference_splits $INFERENCE_SPLITS

echo ""
echo "Model 1 complete!"
echo ""

# Model 2: answerdotai/ModernBERT-base
echo "========================================="
echo "Training Model 2/3: ModernBERT-base"
echo "Hyperparameters: LR=2e-5, BS=8, Epochs=3"
echo "Target ROC-AUC: 0.72"
echo "========================================="
python -m memory_machines.training.classifier.train \
    --data_dir "$DATA_DIR" \
    --model_name "answerdotai/ModernBERT-base" \
    --input_format "$INPUT_FORMAT" \
    --learning_rate 2e-5 \
    --batch_size 8 \
    --epochs 3 \
    --max_length "$MAX_LENGTH" \
    --seed "$SEED" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "modernbert-base-final" \
    --inference_splits $INFERENCE_SPLITS

echo ""
echo "Model 2 complete!"
echo ""

# Model 3: answerdotai/ModernBERT-large
echo "========================================="
echo "Training Model 3/3: ModernBERT-large"
echo "Hyperparameters: LR=5e-6, BS=8, Epochs=2"
echo "Target ROC-AUC: 0.69"
echo "========================================="
python -m memory_machines.training.classifier.train \
    --data_dir "$DATA_DIR" \
    --model_name "answerdotai/ModernBERT-large" \
    --input_format "$INPUT_FORMAT" \
    --learning_rate 5e-6 \
    --batch_size 8 \
    --epochs 2 \
    --max_length "$MAX_LENGTH" \
    --seed "$SEED" \
    --wandb_project "$WANDB_PROJECT" \
    --run_name "modernbert-large-final" \
    --inference_splits $INFERENCE_SPLITS

echo ""
echo "Model 3 complete!"
echo ""

echo "========================================="
echo "All models trained successfully!"
echo "========================================="
echo ""
echo "Results logged to W&B project: $WANDB_PROJECT"
echo "Prediction tables available at: predictions/{split_name}"
echo "Inference metrics available at: inference_metrics/{split_name}"
echo ""
echo "Checkpoints saved to: memory_machines/training/checkpoints/classifier/"
echo "========================================="
