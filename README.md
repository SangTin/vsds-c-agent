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

## v2 — CoT + 10-choice fix

v1 scored **66.52%** accuracy on the public-test leaderboard. v2 adds a prompt
hint that explicitly asks the model to consider every choice on questions with
more than four options, plus an optional two-step chain-of-thought path. The
CoT evaluation was abandoned because it was approximately **17x slower** than
v1.

Enable CoT with `--cot`, or set `model.use_cot: true` in `config.yaml`. Use
`--no-cot` to override an enabled config value. `model.cot_max_tokens` controls
the reasoning budget and defaults to `200`. Because the measured implementation
was approximately 17x slower than v1, it is retained only for comparison and is
not the recommended evaluation path.

Run the full public-test CoT evaluation with:

```bash
bash scripts/eval_full_cot.sh
```

## v3 — RAG (viwiki)

v3 retrieves the top three relevant chunks from a Vietnamese Wikipedia FAISS
index using BGE-m3 dense embeddings, then prepends them to the LLM prompt.
Questions that already contain `Đoạn thông tin:` skip retrieval because their
reading-comprehension passage is already provided.

Install the data/RAG dependencies and build the index once:

```bash
bash scripts/build_viwiki_index.sh
```

The CPU build is expected to take 1-2 hours and is faster on GPU when the
wrapper or `src.rag.build_index` command is adjusted to use `--device cuda`.
`FlagEmbedding` and `faiss-cpu` from `requirements-data.txt` are also required
at runtime whenever RAG is enabled.

Run the full public test with RAG:

```bash
bash scripts/eval_full_rag.sh
```

Expected evaluation time is about 25-30 minutes for 463 questions. Configure
RAG with `rag.enabled`, `rag.top_k`, and `rag.device` in `config.yaml`, or use
the corresponding CLI flags.

Baselines: **v1 = 66.52%**, **v2 = abandoned (17x slow)**, **v3 = dropped**.
See [docs/rag-experiment.md](docs/rag-experiment.md) for the negative-result
writeup and the RAG-to-tools pivot.

## v4 — Tools (Program-of-Thought)

v4 adds an optional Program-of-Thought path for calculation-heavy questions.
The router sends only likely math/quantity questions into this path: LaTeX math,
arithmetic/calculus/economics keywords, units/percent/currency signals, or
mostly numeric choices. Non-math questions continue through the v1 direct MCQ
path.

Enable tools with `--tools`, disable a configured tool run with `--no-tools`,
or set `model.use_tools: true` in `config.yaml`. `--tool-timeout` and
`model.tool_timeout` control the per-snippet subprocess timeout and default to
`5.0` seconds.

When tools are enabled, Qwen first writes a short fenced Python snippet that
prints the final computed value. The snippet runs in an isolated subprocess with
`math`, `sympy`, `itertools`, `fractions`, `decimal`, and `statistics` available.
If the code raises or prints nothing, the error is fed back once and the model
gets one repair attempt. The printed result is then given back to the grammar
constrained MCQ selector. Any tool failure falls back to the normal v1 answer
path, so tools never break batch output.

## Legal RAG pilot (experimental, opt-in)

This pilot tests a different RAG hypothesis from the dropped general Wikipedia
RAG: a small, answer-dense Vietnamese legal corpus retrieved only for law
questions. It is domain-gated, score-thresholded, and independent of `--rag`.
Keep it only if measured net-positive against the tools-only Qwen3.5 baseline.

Prepare the legal corpus and index:

```bash
python scripts/fetch_uts_vlc.py --split 2026
bash scripts/build_legal_index.sh
```

Run with:

```bash
python entrypoint.py \
  --data-dir ../data \
  --output-dir ../output \
  --tools \
  --legal-rag \
  --legal-min-score 0.0 \
  --model-path models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf \
  --n-gpu-layers 99 \
  --verbose
```

The corpus is `undertheseanlp/UTS_VLC` split `2026` (about 24M chars). Expect
roughly 16k chunks and about 20 minutes to build on a small GPU. Legal RAG
fires only when `is_law_question` detects a law question and the top legal
retrieval score is at least `--legal-min-score`; otherwise the pipeline answers
directly with no injected context.

## Political-theory RAG pilot (experimental, opt-in)

This second targeted RAG pilot mirrors the legal path with a separate router,
separate FAISS index, and separate CLI/config flags. It targets the five
mandatory Vietnamese university political-theory subjects: Triết học
Mác-Lênin, Kinh tế Chính trị Mác-Lênin, Chủ nghĩa Xã hội Khoa học, Tư tưởng
Hồ Chí Minh, and Lịch sử Đảng CSVN. Keep it only if measured net-positive
against the tools-only Qwen3.5 baseline.

Prepare the hybrid corpus:

```bash
python scripts/fetch_polysci.py
```

`fetch_polysci.py` downloads the two Duy Tan University PDFs with stable direct
URLs into `data_kb/polysci/raw/`: Triết học Mác-Lênin and Kinh tế Chính trị
Mác-Lênin. Manually drop the remaining three PDFs into the same directory:

```text
data_kb/polysci/raw/tu-tuong-ho-chi-minh.pdf
data_kb/polysci/raw/chu-nghia-xa-hoi-khoa-hoc.pdf
data_kb/polysci/raw/lich-su-dang-cong-san-vn.pdf
```

Chunk every PDF in the raw directory, then build the separate polysci index:

```bash
python scripts/build_targeted_corpus.py \
  --raw data_kb/polysci/raw \
  --out data_kb/polysci/chunks.jsonl
bash scripts/build_polysci_index.sh
```

Run with polysci RAG, optionally alongside legal RAG:

```bash
python entrypoint.py \
  --data-dir ../data \
  --output-dir ../output \
  --tools \
  --legal-rag \
  --polysci-rag \
  --polysci-device cuda \
  --model-path models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf \
  --n-gpu-layers 99 \
  --verbose
```

Polysci RAG fires only after the math and legal routers decline the question.
If `is_polysci_question` detects a political-theory cue and the top retrieval
score is at least `--polysci-min-score`, the pipeline injects textbook context;
otherwise it answers directly with no injected context.

## Docker (submission)

Submission config: **Qwen3.5-9B-Instruct Q4_K_M GGUF + Program-of-Thought
tools**, scored **82.29%** on the public leaderboard. RAG is deliberately
excluded from the Docker image because the experiment was net-negative: the
image does not install `FlagEmbedding`, `faiss`, `torch`,
`sentence-transformers`, or bake `data_kb/` / FAISS indexes. The `--rag` code
path remains in the repository but is off by default and must not be used for
the submission container.

One-time GGUF download before building:

```bash
pip install -U huggingface_hub
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf --local-dir models/qwen3.5-9b
```

Build the offline submission image:

```bash
bash scripts/docker_build.sh
```

Run with GPU:

```bash
bash scripts/docker_run.sh
```

Offline CPU smoke test:

```bash
docker run --rm --network none \
  -v "$PWD/../data":/data \
  -v "$PWD/../output":/output \
  vsds-bangc:latest
```

The image auto-selects GPU offload when `/dev/nvidia0` or `nvidia-smi` is
visible, otherwise it runs on CPU. Override with `--n-gpu-layers N` or
`BANGC_N_GPU_LAYERS=N`. Determinism is preserved with `seed=42`,
`temperature=0`, and the local sympy sandbox, satisfying the BTC requirement
for "chay >=3 lan on dinh".
