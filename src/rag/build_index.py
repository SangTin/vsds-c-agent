"""Build the fp32 FAISS Flat index and aligned metadata for viwiki chunks.

Vectors are added directly to IndexFlatIP in 2,000-record buffers. This keeps
the 6 GB JSONL corpus and the full embedding matrix out of Python heap memory;
FAISS owns the final fp32 matrix as buffers are added.

Periodic checkpointing (--checkpoint-interval) writes the in-progress index and
metadata to the same output paths so a disconnected Colab session can be
resumed with --resume.
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
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Save index+metadata every N processed chunks (0=disabled).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load existing --index-out and --metadata-out and skip already-embedded chunks.",
    )
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


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for raw in f:
            if raw.strip():
                n += 1
    return n


def _truncate_jsonl(path: Path, keep: int) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    kept = 0
    with path.open("r", encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            if kept >= keep:
                break
            dst.write(line)
            kept += 1
    tmp.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    import faiss

    args.index_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)

    resume_offset = 0
    metadata_mode = "w"
    if args.resume and args.index_out.exists() and args.metadata_out.exists():
        index = faiss.read_index(str(args.index_out))
        resume_offset = index.ntotal
        meta_lines = _count_lines(args.metadata_out)
        if meta_lines > resume_offset:
            # Metadata flushes every buffer (2k chunks); index only at checkpoint.
            # A mid-checkpoint disconnect leaves extra metadata lines whose vectors
            # were lost from in-memory. Truncate metadata to match the saved index;
            # those chunks will be re-embedded.
            print(
                f"Metadata has {meta_lines:,} lines but index has {resume_offset:,} "
                f"vectors. Truncating metadata to match (lost "
                f"{meta_lines - resume_offset:,} unsaved chunks; will re-embed).",
                file=sys.stderr,
                flush=True,
            )
            _truncate_jsonl(args.metadata_out, resume_offset)
        elif meta_lines < resume_offset:
            raise ValueError(
                f"Index has {resume_offset} vectors but metadata only "
                f"{meta_lines} lines. Index is ahead of metadata — delete both "
                f"and rebuild from scratch."
            )
        metadata_mode = "a"
        print(
            f"Resuming from checkpoint at {resume_offset:,} chunks",
            file=sys.stderr,
            flush=True,
        )
    else:
        index = faiss.IndexFlatIP(DIMENSION)

    embedder = BGEEmbedder(device=args.device, max_length=args.max_length)

    texts: list[str] = []
    records: list[dict[str, Any]] = []
    processed = resume_offset
    bytes_read = 0
    next_progress = (processed // PROGRESS_INTERVAL + 1) * PROGRESS_INTERVAL
    next_checkpoint = (
        processed + args.checkpoint_interval if args.checkpoint_interval > 0 else 0
    )
    total_bytes = args.chunks.stat().st_size
    started = time.perf_counter()

    def flush(metadata_file: TextIO) -> None:
        nonlocal processed, next_checkpoint
        if not texts:
            return
        vectors = embedder.encode_corpus(texts, batch_size=args.batch_size)
        index.add(vectors)
        for record in records:
            metadata_file.write(json.dumps(record, ensure_ascii=False))
            metadata_file.write("\n")
        processed += len(texts)
        metadata_file.flush()
        texts.clear()
        records.clear()
        if args.checkpoint_interval > 0 and processed >= next_checkpoint:
            ckpt_start = time.perf_counter()
            faiss.write_index(index, str(args.index_out))
            ckpt_elapsed = time.perf_counter() - ckpt_start
            print(
                f"Checkpoint saved at {processed:,} chunks "
                f"(index.ntotal={index.ntotal:,}, wrote in {ckpt_elapsed:.1f}s)",
                file=sys.stderr,
                flush=True,
            )
            next_checkpoint = processed + args.checkpoint_interval

    skipped = 0
    with (
        args.chunks.open("rb") as chunks_file,
        args.metadata_out.open(metadata_mode, encoding="utf-8") as metadata_file,
    ):
        for raw_line in chunks_file:
            bytes_read += len(raw_line)
            if not raw_line.strip():
                continue
            if skipped < resume_offset:
                skipped += 1
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
    new_chunks = processed - resume_offset
    print(
        f"Wrote {processed:,} vectors to {args.index_out} and metadata to "
        f"{args.metadata_out} ({new_chunks:,} new in {elapsed / 60:.1f} min)",
        file=sys.stderr,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    rc = main()
    # FlagEmbedding worker cleanup can hang after a successful long index build.
    os._exit(rc)
