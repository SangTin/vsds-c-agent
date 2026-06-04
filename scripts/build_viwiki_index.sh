#!/usr/bin/env bash
set -euo pipefail

started=$SECONDS

if [[ -f .venv-data/bin/activate ]]; then
  source .venv-data/bin/activate
elif [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

if ! python -c "import FlagEmbedding, faiss" >/dev/null 2>&1; then
  python -m pip install "FlagEmbedding>=1.2" "faiss-cpu>=1.8"
fi

python -m src.rag.build_index \
  --chunks data_kb/viwiki/chunks.jsonl \
  --index-out data_kb/viwiki/index.faiss \
  --metadata-out data_kb/viwiki/metadata.jsonl \
  --device cpu

echo "Index build elapsed seconds: $((SECONDS - started))"
