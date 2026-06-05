#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

python entrypoint.py \
  --data-dir ../data \
  --output-dir ../output \
  --tools \
  --legal-rag \
  --legal-device cuda \
  --model-path models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf \
  --n-gpu-layers 99 \
  --verbose | tee ../output/v6-legal-rag.log
