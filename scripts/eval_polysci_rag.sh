#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p ../output

python entrypoint.py \
  --data-dir ../data \
  --output-dir ../output \
  --tools \
  --legal-rag \
  --legal-device cuda \
  --polysci-rag \
  --polysci-device cuda \
  --model-path models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf \
  --n-gpu-layers 99 \
  --verbose | tee ../output/v7-polysci-rag.log
