#!/usr/bin/env bash
# Build the v6 submission image (v4 + domain-gated legal-RAG).
# Run on a machine WITH internet + GPU + the GGUF + the legal index.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODEL_PATH="models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf"
LEGAL_INDEX="data_kb/legal/index.faiss"
LEGAL_META="data_kb/legal/metadata.jsonl"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing GGUF: $MODEL_PATH" >&2
  echo "  hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir models/qwen3.5-9b" >&2
  exit 1
fi

if [[ ! -f "$LEGAL_INDEX" || ! -f "$LEGAL_META" ]]; then
  echo "Missing legal index ($LEGAL_INDEX / $LEGAL_META)." >&2
  echo "Build it first (needs internet + GPU, ~20 min):" >&2
  echo "  python scripts/fetch_uts_vlc.py" >&2
  echo "  bash scripts/build_legal_index.sh" >&2
  exit 1
fi

docker build -f Dockerfile.v6 -t vsds-bangc:v6 .
echo "Built vsds-bangc:v6"
