#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODEL_PATH="models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing required GGUF model: $MODEL_PATH" >&2
  echo "Download it before building:" >&2
  echo "hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir models/qwen3.5-9b" >&2
  exit 1
fi

docker build -t vsds-bangc:latest .
