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
