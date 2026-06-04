# Bang C Agent

## Overview

This repository implements the Bang C multiple-choice answering pipeline for the Vietnamese Student HackAIthon 2026. Milestone 1 provides validated JSON/CSV input loading, a deterministic dummy answerer, and contract-compliant `pred.csv` output.

## Quickstart

```bash
python entrypoint.py --data-dir ../data --output-dir ../output --verbose
```

## Project Structure

The planned architecture and directory layout are documented in [SPEC_BangC_Agent.md](../SPEC_BangC_Agent.md).

## Milestone Status

| Milestone | Status |
|---|---|
| M0 - Init repo and specification | Done |
| M1 - Framework and I/O | In progress |
| M2-M8 - Model, routing, RAG, agent, optimization, Docker, docs | Pending |

## Reproduce

Install the test dependency with `python -m pip install -r requirements.txt`, run `python -m pytest tests/`, then run the Quickstart command. The generated predictions are written to `../output/pred.csv`.

## Smoke test M2

One-time setup: install `huggingface_hub` CLI, then download both shards of the split GGUF:

```bash
pip install -U huggingface_hub
hf download Qwen/Qwen2.5-7B-Instruct-GGUF \
  qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf \
  --local-dir models/qwen2.5-7b/
```

Smoke test 10 questions:

```bash
bash scripts/eval_subset.sh 10
```

Full public test:

```bash
python entrypoint.py --data-dir ../data --output-dir ../output --model-path models/qwen2.5-7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf --n-gpu-layers 20 --verbose
```

If VRAM runs out, lower `--n-gpu-layers`; try `15`, `10`, `5`, or `0`.
