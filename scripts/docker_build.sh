#!/usr/bin/env bash
# Build the final submission image (v25 = tools + targeted RAG + CoT 600t
# on all questions + alignment-bait shortcut).
# Public-test leaderboard 84.67 (+2.38pp vs v4 82.29).
#
# Assets (Qwen GGUF + legal/polysci FAISS indexes) are pulled from
# HuggingFace Hub at build time, so no local model files are required.
# Override INDEX_REPO with --build-arg if you re-host the indexes elsewhere.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

INDEX_REPO="${INDEX_REPO:-suzueyume/vsds-c-rag-indexes}"

docker build \
  --build-arg INDEX_REPO="$INDEX_REPO" \
  -t vsds-bangc:latest .
echo "Built vsds-bangc:latest (assets pulled from HF Hub)"
