#!/usr/bin/env bash
set -euo pipefail  # Exit on any error / undefined var / pipeline error

echo "🚀 Starting pluckability evaluation runs..."

# Make this script runnable from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ----------- OpenAI -----------
# uv run python -m memory_machines.pluckability.classify \
#     --model "gpt-5.2" \
#     --reasoning-effort "medium" \
#     --instruction-file "$REPO_ROOT/memory_machines/pluckability/instructions/zero_shot.txt" \
#     --max-concurrency 18

# ----------- Anthropic -----------
# NOTE: Need to set the ANTHROPIC_API_KEY environment variable
# Also reasoning effort is not supported, so we need to use the extra-body parameter to enable thinking.
# uv run python -m memory_machines.pluckability.classify \
#     --model "claude-sonnet-4-5" \
#     --instruction-file "$REPO_ROOT/memory_machines/pluckability/instructions/zero_shot.txt" \
#     --base-url "https://api.anthropic.com/v1/" \
#     --auth-token "$ANTHROPIC_API_KEY" \
#     --max-concurrency 18 \
#     --reasoning-effort "medium" \
#     --extra-body '{"thinking": {"type": "enabled", "budget_tokens": 2048}}'

# uv run python -m memory_machines.pluckability.classify \
#     --model "claude-opus-4-5" \
#     --instruction-file "$REPO_ROOT/memory_machines/pluckability/instructions/zero_shot.txt" \
#     --base-url "https://api.anthropic.com/v1/" \
#     --auth-token "$ANTHROPIC_API_KEY" \
#     --max-concurrency 18 \
#     --reasoning-effort "medium" \
#     --extra-body '{"thinking": {"type": "enabled", "budget_tokens": 2048}}'

# uv run python -m memory_machines.pluckability.classify \
#   --model "claude-opus-4-5" \
#   --instruction-file "$REPO_ROOT/memory_machines/pluckability/instructions/few_shot.txt" \
#   --base-url "https://api.anthropic.com/v1/" \
#   --auth-token "$ANTHROPIC_API_KEY" \
#   --max-concurrency 18 \
#   --reasoning-effort "medium" \
#   --extra-body '{"thinking": {"type": "enabled", "budget_tokens": 2048}}'

# ----------- Google -----------
# NOTE: Need to set the GEMINI_API_KEY environment variable
# Gemini supports the reasoning-effort parameter, so we don't need to use the extra-body parameter.
# https://ai.google.dev/gemini-api/docs/openai#thinking
# uv run python -m memory_machines.pluckability.classify \
#     --model "gemini-2.5-pro" \
#     --instruction-file "$REPO_ROOT/memory_machines/pluckability/instructions/few_shot.txt" \
#     --base-url "https://generativelanguage.googleapis.com/v1beta/openai/" \
#     --auth-token "$GEMINI_API_KEY" \
#     --reasoning-effort "medium" \
#     --max-concurrency 18

# ----------- Open Source (via Cerebras) -----------
# Requires CEREBRAS_API_KEY environment variable to be set
# uv run python -m memory_machines.pluckability.classify \
#     --model "gpt-oss-120b" \
#     --instruction-file "$REPO_ROOT/memory_machines/pluckability/instructions/zero_shot.txt" \
#     --base-url "https://api.cerebras.ai/v1" \
#     --auth-token "$CEREBRAS_API_KEY" \
#     --reasoning-effort "medium" \
#     --max-concurrency 8

# NOTE: reasoning effort is not supported for Qwen, so we need to use the extra-body parameter to enable thinking.
# uv run python -m memory_machines.pluckability.classify \
#     --model "qwen-3-32b" \
#     --instruction-file "$REPO_ROOT/memory_machines/pluckability/instructions/zero_shot.txt" \
#     --base-url "https://api.cerebras.ai/v1" \
#     --auth-token "$CEREBRAS_API_KEY" \
#     --max-concurrency 8