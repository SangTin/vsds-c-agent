#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

mkdir -p "$PWD/../output"

echo "CPU-only host: drop --gpus all from this docker run command; the image still works."

docker run --rm --gpus all \
  -v "$PWD/../data":/data \
  -v "$PWD/../output":/output \
  vsds-bangc:latest "$@"
