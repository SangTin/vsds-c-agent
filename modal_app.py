"""Run the legal-RAG pilot eval on a Modal serverless GPU (free-tier friendly).

Replaces renting a GPU box: defines a CUDA image with our deps, caches the
Qwen3.5-9B GGUF + the UTS_VLC legal FAISS index in a Modal Volume, then runs the
full 463-question public test with --tools --legal-rag and writes the predictions
back to the local ../output/ directory for comparison against the 82.29% v4 run.

Usage (from bang-c-agent/, after `modal setup`):
    modal run modal_app.py                 # build assets if needed, then eval
    modal run modal_app.py --rebuild-index # force-rebuild the legal index

Dev/eval only — the BTC submission artifact remains the offline Docker image.
"""

from __future__ import annotations

import modal

MODEL_REPO = "unsloth/Qwen3.5-9B-GGUF"
MODEL_FILE = "Qwen3.5-9B-Q4_K_M.gguf"
GPU = "A10G"  # 24GB — Qwen3.5-9B full offload; cheap on free credit

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "FlagEmbedding==1.2.10",
        "transformers==4.46.3",
        "tokenizers>=0.20,<0.21",
        "accelerate==1.1.1",
        "faiss-cpu>=1.8",
        "datasets>=2.18",
        "pyyaml",
        "huggingface_hub<1.0",
    )
    # Prebuilt CUDA 12.4 wheel — no source build (avoids needing a C/C++ toolchain
    # in the image); this wheel reports gpu_offload=True on the same model locally.
    .pip_install(
        "llama-cpp-python",
        extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cu124",
    )
    # Ship the repo code into the image (latest local state).
    .add_local_dir(".", "/app", ignore=["data_kb", "models", ".venv*", ".git", "results"])
)

app = modal.App("vsds-legal-rag")
cache = modal.Volume.from_name("vsds-cache", create_if_missing=True)
CACHE = "/cache"


@app.function(image=image, gpu=GPU, volumes={CACHE: cache}, timeout=3600)
def ensure_assets(rebuild_index: bool = False) -> dict:
    """Download the GGUF and build the legal FAISS index into the Volume (idempotent)."""
    import os
    import subprocess
    import sys

    os.chdir("/app")
    sys.path.insert(0, "/app")
    model_dir = f"{CACHE}/qwen3.5-9b"
    legal_dir = f"{CACHE}/legal"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(legal_dir, exist_ok=True)

    model_path = f"{model_dir}/{MODEL_FILE}"
    if not os.path.exists(model_path):
        from huggingface_hub import hf_hub_download

        hf_hub_download(MODEL_REPO, MODEL_FILE, local_dir=model_dir)

    index_path = f"{legal_dir}/index.faiss"
    if rebuild_index or not os.path.exists(index_path):
        chunks = f"{legal_dir}/chunks.jsonl"
        subprocess.run(
            [sys.executable, "scripts/fetch_uts_vlc.py", "--out", chunks],
            check=True,
        )
        subprocess.run(
            [
                sys.executable, "-m", "src.rag.build_index",
                "--chunks", chunks,
                "--index-out", index_path,
                "--metadata-out", f"{legal_dir}/metadata.jsonl",
                "--device", "cuda", "--batch-size", "128", "--max-length", "512",
            ],
            check=True,
        )
    cache.commit()
    return {"model": model_path, "index": index_path}


@app.function(image=image, gpu=GPU, volumes={CACHE: cache}, timeout=3600)
def run_eval(legal_rag: bool = True) -> str:
    """Run the full 463-question public test; return pred.csv content."""
    import os
    import subprocess
    import sys

    os.chdir("/app")
    model_path = f"{CACHE}/qwen3.5-9b/{MODEL_FILE}"
    os.makedirs("/tmp/data", exist_ok=True)
    os.makedirs("/tmp/out", exist_ok=True)
    subprocess.run(
        ["cp", "sample_data/public-test_1780368312.json", "/tmp/data/public_test.json"],
        check=True,
    )
    cmd = [
        sys.executable, "entrypoint.py",
        "--data-dir", "/tmp/data", "--output-dir", "/tmp/out",
        "--model-path", model_path, "--n-gpu-layers", "99",
        "--tools", "--verbose",
    ]
    if legal_rag:
        cmd += [
            "--legal-rag", "--legal-device", "cuda",
            "--legal-index", f"{CACHE}/legal/index.faiss",
            "--legal-metadata", f"{CACHE}/legal/metadata.jsonl",
        ]
    subprocess.run(cmd, check=True)
    with open("/tmp/out/pred.csv", encoding="utf-8") as f:
        return f.read()


@app.local_entrypoint()
def main(rebuild_index: bool = False):
    from pathlib import Path

    assets = ensure_assets.remote(rebuild_index=rebuild_index)
    print("assets ready:", assets)
    pred = run_eval.remote(legal_rag=True)
    out = Path("../output/pred-v6-legal-rag-qwen35.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pred, encoding="utf-8")
    rows = pred.strip().count("\n")
    print(f"wrote {out} ({rows} data rows)")
