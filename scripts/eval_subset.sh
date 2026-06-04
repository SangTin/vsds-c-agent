#!/usr/bin/env bash
set -euo pipefail

N="${1:-10}"
SOURCE="../data/public-test_1780368312.json"
SUBSET="/tmp/vsds-subset.json"
INPUT="/tmp/public_test.json"
OUTPUT="/tmp/vsds-out"

python -c 'import json, sys; data=json.load(open(sys.argv[1], encoding="utf-8")); json.dump(data[:int(sys.argv[2])], open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False)' "$SOURCE" "$N" "$SUBSET"
cp "$SUBSET" "$INPUT"
rm -rf "$OUTPUT"

STARTED="$(python -c 'import time; print(time.perf_counter())')"
python entrypoint.py \
  --data-dir /tmp/ \
  --output-dir "$OUTPUT" \
  --model-path models/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  --n-gpu-layers 20 \
  --verbose

python -c 'import sys, time; elapsed=time.perf_counter()-float(sys.argv[1]); n=int(sys.argv[2]); print(f"Script elapsed seconds: {elapsed:.3f}"); print(f"Script average seconds per question: {elapsed / n if n else 0.0:.6f}")' "$STARTED" "$N"
