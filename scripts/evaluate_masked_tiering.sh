#!/usr/bin/env bash
set -euo pipefail  # Exit on any error / undefined var / pipeline error

echo "🚀 Starting tiering evaluation runs..."

# Make this script runnable from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"


# ----------- OpenAI -----------
# uv run python -m memory_machines.tiering.eval_masked_task_tiering \
#     --model "gpt-5.2" \
#     --reasoning-effort "medium"

# ----------- Anthropic -----------
# NOTE: Need to set the ANTHROPIC_API_KEY environment variable
# (w/o internal heldout set)
# uv run python -m memory_machines.tiering.eval_masked_task_tiering \
#     --model "claude-sonnet-4-5" \
#     --base-url "https://api.anthropic.com/v1/" \
#     --auth-token "$ANTHROPIC_API_KEY" \
#     --extra-body '{"thinking": {"type": "enabled", "budget_tokens": 2048}}'

# uv run python -m memory_machines.tiering.eval_masked_task_tiering \
#     --model "claude-sonnet-4-5" \
#     --base-url "https://api.anthropic.com/v1/" \
#     --auth-token "$ANTHROPIC_API_KEY" \
#     --heldout "/path/to/your/heldout_dataset" \
#     --extra-body '{"thinking": {"type": "enabled", "budget_tokens": 2048}}'

# uv run python -m memory_machines.tiering.eval_masked_task_tiering \
#     --model "claude-sonnet-4-6" \
#     --base-url "https://api.anthropic.com/v1/" \
#     --auth-token "$ANTHROPIC_API_KEY" \
#     --heldout "/path/to/your/heldout_dataset" \
#     --extra-body '{"thinking": {"type": "enabled", "budget_tokens": 2048}}'

# uv run python -m memory_machines.tiering.eval_masked_task_tiering \
#     --model "claude-opus-4-5" \
#     --base-url "https://api.anthropic.com/v1/" \
#     --auth-token "$ANTHROPIC_API_KEY" \
#     --heldout "/path/to/your/heldout_dataset" \
#     --extra-body '{"thinking": {"type": "enabled", "budget_tokens": 2048}}'

# ----------- Google -----------
# NOTE: Need to set the GEMINI_API_KEY environment variable
# uv run python -m memory_machines.tiering.eval_masked_task_tiering \
#     --model "gemini-2.5-pro" \
#     --base-url "https://generativelanguage.googleapis.com/v1beta/openai/" \
#     --auth-token "$GEMINI_API_KEY" \
#     --reasoning-effort "medium"

# ----------- Open Source (via Cerebras) -----------
# Requires CEREBRAS_API_KEY environment variable to be set
# uv run python -m memory_machines.tiering.eval_masked_task_tiering \
#     --model "gpt-oss-120b" \
#     --base-url "https://api.cerebras.ai/v1" \
#     --auth-token "$CEREBRAS_API_KEY"

# uv run python -m memory_machines.tiering.eval_masked_task_tiering \
#     --model "qwen-3-32b" \
#     --base-url "https://api.cerebras.ai/v1" \
#     --auth-token "$CEREBRAS_API_KEY"

# ----------- SFT'd Adapter (via W&B) -----------
# uv run python -m memory_machines.tiering.eval_masked_task_tiering \
#     --model "wandb-artifact:///<your-entity>/memory-machines-sft-tiering/sft-tiering-adapter-lora-8:v1" \
#     --base-url "https://api.inference.wandb.ai/v1" \
#     --auth-token "$WB_API_KEY" \
#     --use-train-prompt \
#     --heldout "/path/to/your/heldout_dataset"

echo "✅ All commands are commented out. Uncomment the models you want to run."
