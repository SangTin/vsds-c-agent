"""Stream the Vietnamese Wikipedia dump and emit chunked JSONL for RAG.

Run on WSL (or any box with ~10 GB free disk and internet):
    pip install -r requirements-data.txt
    python scripts/build_viwiki_corpus.py --out data_kb/viwiki/chunks.jsonl

The HF dataset config "20231101.vi" gives ~1.5M articles already cleaned of
wiki-markup. We split each article into ~1500-char chunks on paragraph
boundaries so that an 8K-context LLM can take 3-5 chunks at once.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CHUNK_TARGET_CHARS = 1500
CHUNK_MIN_CHARS = 200


def split_article(text: str, target: int = CHUNK_TARGET_CHARS) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for para in paragraphs:
        if buf_len + len(para) > target and buf:
            chunks.append("\n\n".join(buf))
            buf = [para]
            buf_len = len(para)
        else:
            buf.append(para)
            buf_len += len(para) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return [c for c in chunks if len(c) >= CHUNK_MIN_CHARS]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output JSONL path")
    parser.add_argument("--limit", type=int, default=0, help="cap article count (0 = all)")
    parser.add_argument("--config", default="20231101.vi", help="HF wikipedia config")
    args = parser.parse_args()

    from datasets import load_dataset
    from tqdm import tqdm

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("wikimedia/wikipedia", args.config, split="train", streaming=True)

    written = 0
    with args.out.open("w", encoding="utf-8") as f:
        for i, article in enumerate(tqdm(ds, desc="articles", unit="art")):
            if args.limit and i >= args.limit:
                break
            for chunk_id, text in enumerate(split_article(article["text"])):
                f.write(
                    json.dumps(
                        {
                            "doc_id": article["id"],
                            "title": article["title"],
                            "chunk_id": chunk_id,
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                )
                f.write("\n")
                written += 1

    print(f"wrote {written} chunks to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
