#!/usr/bin/env bash
set -euo pipefail  # Exit on any error / undefined var / pipeline error

echo "🚀 Starting forced-choice evaluation runs..."

# Make this script runnable from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"


# ----------- OpenAI -----------
# uv run python -m memory_machines.forced_choice.choose_most_preferred \
#     --model "gpt-5.2" \
#     --instruction-file "$REPO_ROOT/memory_machines/forced_choice/instructions/simple.txt" \
#     --max-concurrency 18

# ----------- Anthropic -----------
# NOTE: Need to set the ANTHROPIC_API_KEY environment variable
# uv run python -m memory_machines.forced_choice.choose_most_preferred \
#     --model "claude-sonnet-4-5" \
#     --instruction-file "$REPO_ROOT/memory_machines/forced_choice/instructions/simple.txt" \
#     --base-url "https://api.anthropic.com/v1/" \
#     --auth-token "$ANTHROPIC_API_KEY" \
#     --max-concurrency 18

# uv run python -m memory_machines.forced_choice.choose_most_preferred \
#     --model "claude-opus-4-5" \
#     --instruction-file "$REPO_ROOT/memory_machines/forced_choice/instructions/simple.txt" \
#     --base-url "https://api.anthropic.com/v1/" \
#     --auth-token "$ANTHROPIC_API_KEY" \
#     --max-concurrency 18

# uv run python -m memory_machines.forced_choice.choose_most_preferred \
#     --model "claude-opus-4-5" \
#     --instruction-file "$REPO_ROOT/memory_machines/forced_choice/instructions/few_shot.txt" \
#     --base-url "https://api.anthropic.com/v1/" \
#     --auth-token "$ANTHROPIC_API_KEY" \
#     --max-concurrency 18

# ----------- Google -----------
# NOTE: Need to set the GEMINI_API_KEY environment variable
# uv run python -m memory_machines.forced_choice.choose_most_preferred \
#     --model "gemini-2.5-pro" \
#     --instruction-file "$REPO_ROOT/memory_machines/forced_choice/instructions/simple.txt" \
#     --base-url "https://generativelanguage.googleapis.com/v1beta/openai/" \
#     --auth-token "$GEMINI_API_KEY" \
#     --max-concurrency 18

# ----------- Open Source (via Cerebras) -----------
# Requires CEREBRAS_API_KEY environment variable to be set
# uv run python -m memory_machines.forced_choice.choose_most_preferred \
#     --model "gpt-oss-120b" \
#     --instruction-file "$REPO_ROOT/memory_machines/forced_choice/instructions/simple.txt" \
#     --base-url "https://api.cerebras.ai/v1" \
#     --auth-token "$CEREBRAS_API_KEY" \
#     --max-concurrency 8

# uv run python -m memory_machines.forced_choice.choose_most_preferred \
#     --model "qwen-3-32b" \
#     --instruction-file "$REPO_ROOT/memory_machines/forced_choice/instructions/few_shot.txt" \
#     --base-url "https://api.cerebras.ai/v1" \
#     --auth-token "$CEREBRAS_API_KEY" \
#     --max-concurrency 8

echo "✅ All commands are commented out. Uncomment the models you want to run."
