#!/usr/bin/env bash
set -euo pipefail

python entrypoint.py \
  --data-dir ../data \
  --output-dir ../output \
  --model-path models/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --n-gpu-layers 20 \
  --rag \
  --verbose | tee ../output/v3-rag.log
