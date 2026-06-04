"""Build the fp32 FAISS Flat index and aligned metadata for viwiki chunks.

Vectors are added directly to IndexFlatIP in 2,000-record buffers. This keeps
the 6 GB JSONL corpus and the full embedding matrix out of Python heap memory;
FAISS owns the final fp32 matrix as buffers are added.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence, TextIO

from src.rag.embedder import BGEEmbedder

BUFFER_SIZE = 2000
DIMENSION = 1024
PROGRESS_INTERVAL = 50_000


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data_kb/viwiki/chunks.jsonl"),
    )
    parser.add_argument(
        "--index-out",
        type=Path,
        default=Path("data_kb/viwiki/index.faiss"),
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=Path("data_kb/viwiki/metadata.jsonl"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def _metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": record["doc_id"],
        "title": record["title"],
        "chunk_id": record["chunk_id"],
        "text": record["text"],
    }


def _progress(processed: int, bytes_read: int, total_bytes: int, started: float) -> None:
    elapsed = time.perf_counter() - started
    fraction = bytes_read / total_bytes if total_bytes else 1.0
    eta = elapsed * (1.0 - fraction) / fraction if fraction else 0.0
    print(
        f"Indexed {processed:,} chunks | elapsed {elapsed / 60:.1f} min | "
        f"ETA {eta / 60:.1f} min",
        file=sys.stderr,
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    # These heavy dependencies are needed only by the offline index build.
    import faiss

    args.index_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)

    embedder = BGEEmbedder(device=args.device)
    index = faiss.IndexFlatIP(DIMENSION)
    texts: list[str] = []
    records: list[dict[str, Any]] = []
    processed = 0
    bytes_read = 0
    next_progress = PROGRESS_INTERVAL
    total_bytes = args.chunks.stat().st_size
    started = time.perf_counter()

    def flush(metadata_file: TextIO) -> None:
        nonlocal processed
        if not texts:
            return
        vectors = embedder.encode_corpus(texts, batch_size=args.batch_size)
        index.add(vectors)
        for record in records:
            metadata_file.write(json.dumps(record, ensure_ascii=False))
            metadata_file.write("\n")
        processed += len(texts)
        texts.clear()
        records.clear()

    with (
        args.chunks.open("rb") as chunks_file,
        args.metadata_out.open("w", encoding="utf-8") as metadata_file,
    ):
        for raw_line in chunks_file:
            bytes_read += len(raw_line)
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            texts.append(record["text"])
            records.append(_metadata(record))
            if len(texts) >= BUFFER_SIZE:
                flush(metadata_file)
                while processed >= next_progress:
                    _progress(processed, bytes_read, total_bytes, started)
                    next_progress += PROGRESS_INTERVAL
        flush(metadata_file)

    faiss.write_index(index, str(args.index_out))
    elapsed = time.perf_counter() - started
    print(
        f"Wrote {processed:,} vectors to {args.index_out} and metadata to "
        f"{args.metadata_out} in {elapsed / 60:.1f} min",
        file=sys.stderr,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    rc = main()
    # FlagEmbedding worker cleanup can hang after a successful long index build.
    os._exit(rc)
