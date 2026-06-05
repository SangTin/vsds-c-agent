"""Fetch and chunk the undertheseanlp/UTS_VLC Vietnamese legal corpus.

The output schema matches src.rag.build_index:
    {"doc_id": str, "title": str, "chunk_id": int, "text": str}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

try:
    from scripts.build_targeted_corpus import slugify, split_text
except ModuleNotFoundError:
    from build_targeted_corpus import slugify, split_text


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="undertheseanlp/UTS_VLC")
    parser.add_argument("--split", default="2026")
    parser.add_argument("--out", type=Path, default=Path("data_kb/legal/chunks.jsonl"))
    parser.add_argument(
        "--types",
        nargs="+",
        choices=("code", "law", "constitution"),
        default=None,
        help="Optional document types to keep. Defaults to all types.",
    )
    return parser.parse_args(argv)


def _load_dataset(name: str, split: str) -> Any:
    from datasets import load_dataset

    return load_dataset(name, split=split)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    allowed_types = set(args.types) if args.types else None
    dataset = _load_dataset(args.dataset, args.split)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    docs_seen = 0
    docs_written = 0
    chunks_written = 0
    with args.out.open("w", encoding="utf-8") as file:
        for row in dataset:
            docs_seen += 1
            doc_type = str(row.get("type", ""))
            if allowed_types is not None and doc_type not in allowed_types:
                continue

            content = str(row.get("content") or "")
            doc_id = str(
                row.get("id")
                or slugify(str(row.get("filename") or row.get("title") or docs_seen))
            )
            title = str(row.get("title") or row.get("filename") or doc_id)
            chunks = split_text(content)
            for chunk_id, chunk in enumerate(chunks):
                file.write(
                    json.dumps(
                        {
                            "doc_id": doc_id,
                            "title": title,
                            "chunk_id": chunk_id,
                            "text": chunk,
                        },
                        ensure_ascii=False,
                    )
                )
                file.write("\n")
            docs_written += 1
            chunks_written += len(chunks)

    print(
        f"done: {docs_written} documents kept from {docs_seen} scanned -> "
        f"{chunks_written} chunks at {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
