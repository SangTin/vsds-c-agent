"""Upload built FAISS indexes (legal + polysci) to a HuggingFace Hub dataset.

Run this ONCE locally on the machine that already has the indexes built:
    data_kb/legal/{index.faiss, metadata.jsonl}
    data_kb/polysci/{index.faiss, metadata.jsonl}

Usage:
    export HF_TOKEN=hf_xxx     # write-enabled token
    python scripts/upload_rag_indexes.py --repo <HF_USER>/vsds-c-rag-indexes

The Dockerfile.v10 build then pulls these files via hf_hub_download at
build time, so the offline runtime image bakes them in and GitHub Actions
(or any other CI) can build without the user having to upload assets by
hand each time.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="<user>/<dataset-repo>")
    parser.add_argument("--legal-dir", default="data_kb/legal")
    parser.add_argument("--polysci-dir", default="data_kb/polysci")
    parser.add_argument(
        "--private",
        action="store_true",
        help="create as private dataset (default public so the Docker "
        "build does not need a token)",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: set HF_TOKEN env var to a write-enabled token", file=sys.stderr)
        return 2

    files = [
        (Path(args.legal_dir) / "index.faiss", "legal/index.faiss"),
        (Path(args.legal_dir) / "metadata.jsonl", "legal/metadata.jsonl"),
        (Path(args.polysci_dir) / "index.faiss", "polysci/index.faiss"),
        (Path(args.polysci_dir) / "metadata.jsonl", "polysci/metadata.jsonl"),
    ]
    missing = [str(src) for src, _ in files if not src.is_file()]
    if missing:
        print("ERROR: missing required files:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        print(
            "Build them first with:\n"
            "  python scripts/fetch_uts_vlc.py && bash scripts/build_legal_index.sh\n"
            "  python scripts/fetch_polysci.py + drop the 2 user PDFs into data_kb/polysci/raw\n"
            "  python scripts/build_targeted_corpus.py --raw data_kb/polysci/raw "
            "--out data_kb/polysci/chunks.jsonl\n"
            "  bash scripts/build_polysci_index.sh",
            file=sys.stderr,
        )
        return 2

    api = HfApi(token=token)
    create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True, token=token)

    for src, dest in files:
        size_mb = src.stat().st_size / (1024 * 1024)
        print(f"uploading {src} -> {args.repo}:{dest} ({size_mb:.1f} MB)")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=dest,
            repo_id=args.repo,
            repo_type="dataset",
        )

    print(f"done. Verify at https://huggingface.co/datasets/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
