#!/usr/bin/env bash
set -euo pipefail

mkdir -p ../output

{
  time python entrypoint.py \
    --data-dir ../data \
    --output-dir ../output \
    --model-path models/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
    --n-gpu-layers 20 \
    --cot \
    --verbose
} 2>&1 | tee ../output/v2-cot.log

printf '\npred.csv head:\n'
head ../output/pred.csv
printf '\npred.csv tail:\n'
tail ../output/pred.csv
