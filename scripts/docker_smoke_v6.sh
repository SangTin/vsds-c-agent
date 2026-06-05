#!/usr/bin/env bash
# Offline smoke test for the v6 image: runs with --network none on a tiny data dir
# and verifies pred.csv is produced with diverse letters (not an all-A stub) and that
# the legal-RAG path loaded without crashing the batch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_DIR="${1:-../data}"
OUT_DIR="$(mktemp -d)"

echo "Running v6 OFFLINE (--network none); data=$DATA_DIR out=$OUT_DIR"
# Drop --gpus all on a CPU-only host; the image still runs (slower).
docker run --rm --network none ${GPUS:+--gpus all} \
  -v "$(cd "$DATA_DIR" && pwd)":/data \
  -v "$OUT_DIR":/output \
  vsds-bangc:v6

PRED="$OUT_DIR/pred.csv"
if [[ ! -f "$PRED" ]]; then
  echo "FAIL: no pred.csv produced" >&2
  exit 1
fi

rows=$(($(wc -l < "$PRED") - 1))
distinct=$(tail -n +2 "$PRED" | cut -d, -f2 | sort -u | wc -l)
echo "pred.csv rows=$rows distinct_letters=$distinct"
if [[ "$distinct" -le 1 ]]; then
  echo "FAIL: predictions collapsed to a single letter (stub fallback)" >&2
  exit 1
fi
echo "PASS: v6 image runs offline and produces diverse predictions"
