#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
started=$SECONDS

if [[ -f .venv-data/bin/activate ]]; then
  source .venv-data/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

python -m src.rag.build_index \
  --chunks data_kb/polysci/chunks.jsonl \
  --index-out data_kb/polysci/index.faiss \
  --metadata-out data_kb/polysci/metadata.jsonl \
  --device cuda \
  --batch-size 64 \
  --max-length 512

echo "Polysci index build elapsed seconds: $((SECONDS - started))"
