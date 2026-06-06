"""Run the targeted-RAG pilot eval on a Modal serverless GPU (free-tier friendly).

Replaces renting a GPU box: defines a CUDA image with our deps, caches the
Qwen3.5-9B GGUF + targeted FAISS indexes in a Modal Volume, then runs the full
463-question public test with tools and targeted RAG flags. Predictions are
written back to the local ../output/ directory for comparison against the 82.29%
v4 run.

Usage (from bang-c-agent/, after `modal setup`):
    modal run modal_app.py                 # build assets if needed, then eval
    modal run modal_app.py --rebuild-index # force-rebuild the legal index

Dev/eval only — the BTC submission artifact remains the offline Docker image.
"""

from __future__ import annotations

import modal

MODEL_REPO = "unsloth/Qwen3.5-9B-GGUF"
# Q4_K_M is the canonical submission quant — empirically tested Q5_K_M ≈ Q4 on this MCQ
# set (Δ proxy = -0.65pp), so stay on Q4 for smaller image and matched-Docker reproducibility.
MODEL_FILE = "Qwen3.5-9B-Q4_K_M.gguf"
GPU = "A10G"  # 24GB — Qwen3.5-9B full offload; cheap on free credit

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11")
    # libgomp1: required by the llama-cpp prebuilt wheel (links libgomp.so.1).
    .apt_install("git", "libgomp1")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "FlagEmbedding==1.2.10",
        "transformers==4.46.3",
        "tokenizers>=0.20,<0.21",
        "accelerate==1.1.1",
        "faiss-cpu>=1.8",
        "datasets>=2.18",
        "pyyaml",
        "pypdf",
        "huggingface_hub<1.0",
    )
    # Prebuilt CUDA 12.4 wheel — no source build (avoids needing a C/C++ toolchain
    # in the image); this wheel reports gpu_offload=True on the same model locally.
    .pip_install(
        "llama-cpp-python",
        extra_index_url="https://abetlen.github.io/llama-cpp-python/whl/cu124",
    )
    # Ship the repo code into the image (latest local state).
    .add_local_dir(
        ".",
        "/app",
        ignore=[
            "data_kb/viwiki",
            "data_kb/legal",
            "models",
            ".venv*",
            ".git",
            "results",
            "*.gguf",
        ],
    )
)

app = modal.App("vsds-legal-rag")
cache = modal.Volume.from_name("vsds-cache", create_if_missing=True)
CACHE = "/cache"


@app.function(image=image, gpu=GPU, volumes={CACHE: cache}, timeout=3600)
def ensure_assets(rebuild_index: bool = False, model_file: str | None = None) -> dict:
    """Download the GGUF and build the legal FAISS index into the Volume (idempotent)."""
    import os
    import subprocess
    import sys

    os.chdir("/app")
    sys.path.insert(0, "/app")
    model_dir = f"{CACHE}/qwen3.5-9b"
    legal_dir = f"{CACHE}/legal"
    polysci_dir = f"{CACHE}/polysci"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(legal_dir, exist_ok=True)
    os.makedirs(polysci_dir, exist_ok=True)

    resolved_model_file = model_file or MODEL_FILE
    model_path = f"{model_dir}/{resolved_model_file}"
    if not os.path.exists(model_path):
        from huggingface_hub import hf_hub_download

        hf_hub_download(MODEL_REPO, resolved_model_file, local_dir=model_dir)

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
    polysci_raw = "/app/data_kb/polysci/raw"
    try:
        polysci_pdfs = [
            name
            for name in os.listdir(polysci_raw)
            if name.lower().endswith(".pdf")
        ]
    except FileNotFoundError:
        polysci_pdfs = []
    polysci_index_path = f"{polysci_dir}/index.faiss"
    polysci_metadata_path = f"{polysci_dir}/metadata.jsonl"
    if polysci_pdfs and (
        rebuild_index
        or not os.path.exists(polysci_index_path)
        or not os.path.exists(polysci_metadata_path)
    ):
        polysci_chunks = f"{polysci_dir}/chunks.jsonl"
        subprocess.run(
            [
                sys.executable,
                "scripts/build_targeted_corpus.py",
                "--raw",
                polysci_raw,
                "--out",
                polysci_chunks,
            ],
            check=True,
        )
        subprocess.run(
            [
                sys.executable, "-m", "src.rag.build_index",
                "--chunks", polysci_chunks,
                "--index-out", polysci_index_path,
                "--metadata-out", polysci_metadata_path,
                "--device", "cuda", "--batch-size", "64", "--max-length", "512",
            ],
            check=True,
        )
    cache.commit()
    return {
        "model": model_path,
        "index": index_path,
        "polysci_index": polysci_index_path if polysci_pdfs else None,
    }


@app.function(image=image, gpu=GPU, volumes={CACHE: cache}, timeout=14400)
def run_eval(
    legal_rag: bool = True,
    polysci_rag: bool = True,
    self_consistency: bool = True,
    cot: bool = False,
    pot_ranking: bool = False,
    cot_max_tokens: int | None = None,
    model_file: str | None = None,
) -> str:
    """Run the full 463-question public test; return pred.csv content."""
    import os
    import subprocess
    import sys

    os.chdir("/app")
    resolved_model_file = model_file or MODEL_FILE
    model_path = f"{CACHE}/qwen3.5-9b/{resolved_model_file}"

    # Fail loud if anything that would silently route to stub: model file missing,
    # llama_cpp not importable, or CUDA offload unavailable on this GPU. A silent
    # stub fallback produces "all-A" predictions (we hit this once already).
    if not os.path.exists(model_path):
        raise RuntimeError(f"model not in volume: {model_path}")
    try:
        import llama_cpp
    except Exception as exc:
        raise RuntimeError(f"llama_cpp import failed in eval container: {exc}") from exc
    if not llama_cpp.llama_supports_gpu_offload():
        raise RuntimeError(
            "llama-cpp wheel reports no GPU offload — would run on CPU and look like stub on time-out."
        )

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
    if self_consistency:
        cmd += ["--self-consistency"]
    if cot:
        cmd += ["--cot"]
    if pot_ranking:
        cmd += ["--pot-ranking"]
    if cot_max_tokens is not None:
        cmd += ["--cot-max-tokens", str(cot_max_tokens)]
    if legal_rag:
        cmd += [
            "--legal-rag", "--legal-device", "cuda",
            "--legal-index", f"{CACHE}/legal/index.faiss",
            "--legal-metadata", f"{CACHE}/legal/metadata.jsonl",
        ]
    if polysci_rag:
        cmd += [
            "--polysci-rag", "--polysci-device", "cuda",
            "--polysci-index", f"{CACHE}/polysci/index.faiss",
            "--polysci-metadata", f"{CACHE}/polysci/metadata.jsonl",
        ]
    subprocess.run(cmd, check=True)
    with open("/tmp/out/pred.csv", encoding="utf-8") as f:
        text = f.read()
    # Sanity: a healthy run on Qwen3.5+tools uses A..J broadly; all-A means stub.
    rows = [line.split(",")[1] for line in text.strip().splitlines()[1:]]
    distinct = set(rows)
    if len(distinct) <= 1:
        raise RuntimeError(
            f"eval collapsed to single letter {distinct} — LLM almost certainly fell to stub; "
            f"check container logs above for 'LLM disabled, falling back to stub: ...'"
        )
    return text


@app.local_entrypoint()
def main(rebuild_index: bool = False):
    """v9: self-consistency only (3-prompt majority vote), no CoT."""
    from pathlib import Path

    assets = ensure_assets.remote(rebuild_index=rebuild_index)
    print("assets ready:", assets)
    pred = run_eval.remote(
        legal_rag=True, polysci_rag=True, self_consistency=True, cot=False
    )
    out = Path("../output/pred-v9-self-consistency.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pred, encoding="utf-8")
    rows = pred.strip().count("\n")
    print(f"wrote {out} ({rows} data rows)")


@app.local_entrypoint()
def main_cot(rebuild_index: bool = False):
    """v10: CoT-then-grammar (4-step reasoning), no self-consistency."""
    from pathlib import Path

    assets = ensure_assets.remote(rebuild_index=rebuild_index)
    print("assets ready:", assets)
    pred = run_eval.remote(
        legal_rag=True, polysci_rag=True, self_consistency=False, cot=True
    )
    out = Path("../output/pred-v10-cot.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pred, encoding="utf-8")
    rows = pred.strip().count("\n")
    print(f"wrote {out} ({rows} data rows)")


@app.local_entrypoint()
def main_cot_long(rebuild_index: bool = False):
    """v13: CoT with extended 600-token reasoning budget (v10 ran at 200)."""
    from pathlib import Path

    assets = ensure_assets.remote(rebuild_index=rebuild_index)
    print("assets ready:", assets)
    pred = run_eval.remote(
        legal_rag=True,
        polysci_rag=True,
        self_consistency=False,
        cot=True,
        cot_max_tokens=600,
    )
    out = Path("../output/pred-v13-cot-long.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pred, encoding="utf-8")
    rows = pred.strip().count("\n")
    print(f"wrote {out} ({rows} data rows)")


@app.local_entrypoint()
def main_cot_long_q5(rebuild_index: bool = False):
    """v18: v13 stack (CoT 600t + RAG) on Q5_K_M quant (vs default Q4_K_M)."""
    from pathlib import Path

    q5_file = "Qwen3.5-9B-Q5_K_M.gguf"
    assets = ensure_assets.remote(rebuild_index=rebuild_index, model_file=q5_file)
    print("assets ready:", assets)
    pred = run_eval.remote(
        legal_rag=True,
        polysci_rag=True,
        self_consistency=False,
        cot=True,
        cot_max_tokens=600,
        model_file=q5_file,
    )
    out = Path("../output/pred-v18-cot-long-q5.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pred, encoding="utf-8")
    rows = pred.strip().count("\n")
    print(f"wrote {out} ({rows} data rows)")


@app.local_entrypoint()
def main_pot_ranking(rebuild_index: bool = False):
    """v16: v10 baseline + PoT answer-ranking for math."""
    from pathlib import Path

    assets = ensure_assets.remote(rebuild_index=rebuild_index)
    print("assets ready:", assets)
    pred = run_eval.remote(
        legal_rag=True,
        polysci_rag=True,
        self_consistency=False,
        cot=True,
        pot_ranking=True,
    )
    out = Path("../output/pred-v16-pot-ranking.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pred, encoding="utf-8")
    rows = pred.strip().count("\n")
    print(f"wrote {out} ({rows} data rows)")


@app.local_entrypoint()
def main_combined(rebuild_index: bool = False):
    """v11: CoT + self-consistency combined (1 reasoning + 3 grammar extractions)."""
    from pathlib import Path

    assets = ensure_assets.remote(rebuild_index=rebuild_index)
    print("assets ready:", assets)
    pred = run_eval.remote(
        legal_rag=True, polysci_rag=True, self_consistency=True, cot=True
    )
    out = Path("../output/pred-v11-cot-sc.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(pred, encoding="utf-8")
    rows = pred.strip().count("\n")
    print(f"wrote {out} ({rows} data rows)")


@app.local_entrypoint()
def det_check():
    """Run the full eval twice; assert byte-identical predictions.

    BTC requires the pipeline to run >=3 times with stable output. PoT was
    already verified deterministic; this confirms the legal-RAG path
    (BGE-m3 GPU encode + FAISS + context-augmented selection) also is.
    """
    from pathlib import Path

    ensure_assets.remote()
    run1 = run_eval.remote(legal_rag=True)
    run2 = run_eval.remote(legal_rag=True)
    out_dir = Path("../output")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pred-det1.csv").write_text(run1, encoding="utf-8")
    (out_dir / "pred-det2.csv").write_text(run2, encoding="utf-8")
    if run1 == run2:
        print("DETERMINISTIC: 2 runs byte-identical")
    else:
        diffs = sum(1 for a, b in zip(run1.splitlines(), run2.splitlines()) if a != b)
        print(f"NON-DETERMINISTIC: {diffs} differing lines between pred-det1.csv and pred-det2.csv")
