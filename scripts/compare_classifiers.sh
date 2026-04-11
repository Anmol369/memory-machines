#!/usr/bin/env bash
# Compare DSPy program vs tiering classifier performance
#
# Usage: ./scripts/compare_classifiers.sh <program_path>
# Example: ./scripts/compare_classifiers.sh memory_machines/training/checkpoints/optimized_tiering/20251222_133944/optimized_program.json

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <program_path>"
    echo ""
    echo "Arguments:"
    echo "  program_path   Path to DSPy program JSON file"
    echo ""
    echo "Example:"
    echo "  $0 memory_machines/training/checkpoints/optimized_tiering/20251222_133944/optimized_program.json"
    exit 1
fi

PROGRAM_PATH="$1"

# Hardcoded configuration (matches train.py config)
DSPY_MODEL="anthropic/claude-sonnet-4-5"
TIERING_MODEL="claude-sonnet-4-5"
DATASET="laddermedia/srs-highlights"
SPLIT="grounded"
OUTPUT_DIR="memory_machines/training/results/classifier_comparison"

echo "========================================="
echo "Classifier Comparison"
echo "========================================="
echo "DSPy Program: $PROGRAM_PATH"
echo "DSPy Model: $DSPY_MODEL (same as train.py)"
echo "Tiering Model: $TIERING_MODEL"
echo "Dataset: $DATASET (split: $SPLIT)"
echo "Output Directory: $OUTPUT_DIR"
echo "========================================="
echo ""

# Run comparison
uv run python -m memory_machines.training.optimized_tiering.compare_classifiers \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --auth-token "$ANTHROPIC_API_KEY" \
    --program-path "$PROGRAM_PATH" \
    --model "$TIERING_MODEL" \
    --dspy-model "$DSPY_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --extra-body '{"thinking": {"type": "enabled", "budget_tokens": 2048}}'

echo ""
echo "========================================="
echo "Results saved to: $OUTPUT_DIR/comparison_$TIERING_MODEL.jsonl"
echo "========================================="
