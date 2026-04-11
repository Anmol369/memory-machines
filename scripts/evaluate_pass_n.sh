#!/usr/bin/env bash
set -euo pipefail  # Exit on any error / undefined var / pipeline error

echo "🚀 Starting pass@N evaluation runs..."

# Make this script runnable from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ----------- Open Source Models (via Cerebras) -----------
# Requires CEREBRAS_API_KEY environment variable to be set
# These models are used to evaluate pass@8 accuracy for GRPO viability assessment

# qwen-3-235b-a22b-instruct-2507
# uv run python -m memory_machines.tiering.eval_pass_n \
#     --model "qwen-3-235b-a22b-instruct-2507" \
#     --base-url "https://api.cerebras.ai/v1" \
#     --auth-token "$CEREBRAS_API_KEY" \
#     --temperature 1.0 \
#     --n-samples 8 \
#     --split "grounded"

# # GPT-OSS 120B
# uv run python -m memory_machines.tiering.eval_pass_n \
#     --model "gpt-oss-120b" \
#     --base-url "https://api.cerebras.ai/v1" \
#     --auth-token "$CEREBRAS_API_KEY" \
#     --temperature 1.0 \
#     --n-samples 8 \
#     --split "grounded"

# # Qwen3 32B
# uv run python -m memory_machines.tiering.eval_pass_n \
#     --model "qwen-3-32b" \
#     --base-url "https://api.cerebras.ai/v1" \
#     --auth-token "$CEREBRAS_API_KEY" \
#     --temperature 1.0 \
#     --n-samples 8 \
#     --split "grounded"


# DeepSeek V3.1
# Requires TOGETHER_API_KEY environment variable to be set
# uv run python -m memory_machines.tiering.eval_pass_n \
#     --model "deepseek-ai/DeepSeek-V3.1" \
#     --base-url "https://api.together.xyz/v1" \
#     --auth-token "$TOGETHER_API_KEY" \
#     --temperature 1.0 \
#     --n-samples 8 \
#     --split "grounded"

echo "✅ All pass@N evaluations complete!"
