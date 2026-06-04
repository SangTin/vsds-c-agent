#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
python entrypoint.py --data-dir ../data --output-dir ../output --verbose
