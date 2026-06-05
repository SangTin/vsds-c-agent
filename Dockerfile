# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

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
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY entrypoint.py config.yaml ./
COPY src ./src
COPY models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf /app/models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf

ENTRYPOINT ["python3", "entrypoint.py"]
CMD ["--data-dir", "/data", "--output-dir", "/output", "--model-path", "/app/models/qwen3.5-9b/Qwen3.5-9B-Q4_K_M.gguf", "--tools"]
