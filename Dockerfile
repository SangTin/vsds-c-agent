# syntax=docker/dockerfile:1
#
# v25 submission image (public-test leaderboard 84.67 = +2.38pp vs v4 baseline).
#
# Config: Qwen3.5-9B Q4_K_M + Program-of-Thought (sympy) + targeted RAG
# (legal UTS_VLC + 4 polysci textbooks via BGE-m3 + FAISS) + CoT 4-step
# 600-token reasoning (applied on all questions incl. passage) +
# alignment-bait shortcut (HOW-TO + violation + refusal choice → bypass
# LLM with the refusal letter).
#
# Failed variants are preserved on git branches/tags, not in this image:
#   self-consistency (v9 -0.86pp), PoT-ranking (v16 -0.43pp),
#   self-verification (v17 -1.29pp), Q5_K_M quant (v18 -1.08pp).
#
# Build is fully cloud-friendly: pulls the GGUF from unsloth/Qwen3.5-9B-GGUF
# and the two FAISS indexes from a HuggingFace dataset repo (default
# suzueyume/vsds-c-rag-indexes, override with --build-arg INDEX_REPO=...).
# No local model/index files required. Runtime is fully offline
# (HF_HUB_OFFLINE=1, all assets baked into the image).

ARG INDEX_REPO=suzueyume/vsds-c-rag-indexes

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ARG INDEX_REPO
ENV INDEX_REPO=${INDEX_REPO} \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        g++ \
        gcc \
        git \
        ninja-build \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
        python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=all-major" \
    FORCE_CMAKE=1

RUN python -m pip install --upgrade pip setuptools wheel scikit-build-core ninja \
    && python -m pip install --no-cache-dir sympy PyYAML \
    && python -m pip install --no-cache-dir --no-binary=llama-cpp-python llama-cpp-python

# RAG deps. torch is CPU-only on purpose: legal+polysci embedding runs on CPU,
# so we avoid bundling the ~2.5GB CUDA torch wheel.
RUN python -m pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch \
    && python -m pip install --no-cache-dir \
        "FlagEmbedding==1.2.10" \
        "transformers==4.46.3" \
        "tokenizers>=0.20,<0.21" \
        "accelerate==1.1.1" \
        "huggingface-hub<1.0" \
        "faiss-cpu>=1.8"

# Bake BGE-m3 weights into a stable HF cache dir so runtime needs no network.
ENV HF_HOME=/opt/hf
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3')"

# Pull GGUF (public) and the two FAISS indexes (uploaded once via
# scripts/upload_rag_indexes.py) into staging dirs so the runtime stage can
# COPY them at fixed paths.
RUN mkdir -p /opt/assets/models/qwen3.5-9b \
             /opt/assets/data_kb/legal \
             /opt/assets/data_kb/polysci
RUN python <<'PY'
import os
from huggingface_hub import hf_hub_download

INDEX_REPO = os.environ["INDEX_REPO"]

hf_hub_download(
    "unsloth/Qwen3.5-9B-GGUF",
    "Qwen3.5-9B-Q4_K_M.gguf",
    local_dir="/opt/assets/models/qwen3.5-9b",
)

for src in [
    "legal/index.faiss",
    "legal/metadata.jsonl",
    "polysci/index.faiss",
    "polysci/metadata.jsonl",
]:
    hf_hub_download(
        INDEX_REPO,
        src,
        local_dir="/opt/assets/data_kb",
        repo_type="dataset",
    )
PY

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        python3.11 \
    && apt-get purge -y --auto-remove software-properties-common \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf /opt/hf
COPY --from=builder /opt/assets/models /app/models
COPY --from=builder /opt/assets/data_kb /app/data_kb
COPY entrypoint.py config.yaml ./
COPY src ./src

ENTRYPOINT ["python3", "entrypoint.py"]
CMD ["--data-dir", "/data", "--output-dir", "/output", \
     "--model-path", "/app/models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf", \
     "--tools", \
     "--cot", "--cot-max-tokens", "600", \
     "--cot-passage", "--cot-passage-max-chars", "10000", \
     "--alignment-override", \
     "--legal-rag", "--legal-device", "cpu", \
     "--legal-index", "/app/data_kb/legal/index.faiss", \
     "--legal-metadata", "/app/data_kb/legal/metadata.jsonl", \
     "--polysci-rag", "--polysci-device", "cpu", \
     "--polysci-index", "/app/data_kb/polysci/index.faiss", \
     "--polysci-metadata", "/app/data_kb/polysci/metadata.jsonl"]
