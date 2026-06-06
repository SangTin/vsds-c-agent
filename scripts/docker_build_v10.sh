#!/usr/bin/env bash
# Build the v10 submission image (Dockerfile.v10): tools + RAG + CoT.
# Leaderboard 83.37 (+1.08pp vs v4 82.29). Self-consistency intentionally
# omitted (regressed v9 to 81.43). Run on a machine WITH internet + the assets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODEL_PATH="models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf"
LEGAL_INDEX="data_kb/legal/index.faiss"
LEGAL_META="data_kb/legal/metadata.jsonl"
POLYSCI_INDEX="data_kb/polysci/index.faiss"
POLYSCI_META="data_kb/polysci/metadata.jsonl"

missing=0
for f in "$MODEL_PATH" "$LEGAL_INDEX" "$LEGAL_META" "$POLYSCI_INDEX" "$POLYSCI_META"; do
  if [[ ! -f "$f" ]]; then
    echo "Missing: $f" >&2
    missing=1
  fi
done
if [[ $missing -ne 0 ]]; then
  echo >&2
  echo "Prep the assets first:" >&2
  echo "  hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir models/qwen3.5-9b" >&2
  echo "  python scripts/fetch_uts_vlc.py && bash scripts/build_legal_index.sh" >&2
  echo "  python scripts/fetch_polysci.py     # plus drop the 2 user-provided PDFs" >&2
  echo "  python scripts/build_targeted_corpus.py --raw data_kb/polysci/raw --out data_kb/polysci/chunks.jsonl" >&2
  echo "  bash scripts/build_polysci_index.sh" >&2
  exit 1
fi

docker build -f Dockerfile.v10 -t vsds-bangc:v10 .
echo "Built vsds-bangc:v10"
