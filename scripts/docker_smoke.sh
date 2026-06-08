#!/usr/bin/env bash
# Offline smoke test for the v25 submission image: --network none on a
# small data dir, asserts pred.csv has diverse letters (catches stub
# fallback + offline RAG load failures). v25 inference ≈ 2.5x v4 baseline
# (CoT reasoning + grammar extraction + alignment-bait shortcut).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_DIR="${1:-../data}"
OUT_DIR="$(mktemp -d)"

echo "Running v25 OFFLINE (--network none); data=$DATA_DIR out=$OUT_DIR"
docker run --rm --network none ${GPUS:+--gpus all} \
  -v "$(cd "$DATA_DIR" && pwd)":/data \
  -v "$OUT_DIR":/output \
  vsds-bangc:latest

PRED="$OUT_DIR/pred.csv"
if [[ ! -f "$PRED" ]]; then
  echo "FAIL: no pred.csv produced" >&2
  exit 1
fi

rows=$(($(wc -l < "$PRED") - 1))
distinct=$(tail -n +2 "$PRED" | cut -d, -f2 | sort -u | wc -l)
echo "pred.csv rows=$rows distinct_letters=$distinct"
if [[ "$distinct" -le 1 ]]; then
  echo "FAIL: predictions collapsed to a single letter (stub or CoT extraction failure)" >&2
  exit 1
fi
echo "PASS: v25 image runs offline and produces diverse predictions"
