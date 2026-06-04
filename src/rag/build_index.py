"""Build the fp32 FAISS Flat index and aligned metadata for viwiki chunks.

Vectors are added directly to IndexFlatIP in 2,000-record buffers. This keeps
the 6 GB JSONL corpus and the full embedding matrix out of Python heap memory;
FAISS owns the final fp32 matrix as buffers are added.

Periodic checkpointing (--checkpoint-interval) writes the in-progress index and
metadata to the same output paths so a disconnected Colab session can be
resumed with --resume.

When the output paths are on a FUSE-mounted filesystem (Google Drive in Colab),
the hot-path metadata writes can stay in the FUSE buffer for minutes before
syncing to the cloud. A disconnect inside that window silently loses the most
recent checkpoint. This script writes to a local staging directory (--staging-dir,
default /tmp) and only touches the final paths via a copy+fsync+atomic rename
at checkpoint boundaries.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("/tmp"),
        help="Local-disk directory used as a write buffer before copying to "
        "--index-out / --metadata-out. Bypasses FUSE buffering on Drive mounts.",
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


def _sync_to_drive(local_path: Path, remote_path: Path) -> float:
    """Copy a local file to a FUSE-mounted path with fsync + atomic rename.

    Returns elapsed seconds. Skips work if the local file is missing.
    """
    if not local_path.exists():
        return 0.0
    start = time.perf_counter()
    remote_path.parent.mkdir(parents=True, exist_ok=True)
    remote_tmp = remote_path.with_suffix(remote_path.suffix + ".part")
    shutil.copyfile(local_path, remote_tmp)
    try:
        with remote_tmp.open("rb") as f:
            os.fsync(f.fileno())
    except OSError:
        pass  # FUSE may not honor fsync; rename below still forces visibility
    os.replace(remote_tmp, remote_path)
    return time.perf_counter() - start


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    import faiss

    args.index_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    args.staging_dir.mkdir(parents=True, exist_ok=True)

    staging_index = args.staging_dir / args.index_out.name
    staging_metadata = args.staging_dir / args.metadata_out.name

    resume_offset = 0
    metadata_mode = "w"
    if args.resume and args.index_out.exists() and args.metadata_out.exists():
        # Pull both files into staging so the run is decoupled from the FUSE mount.
        shutil.copyfile(args.index_out, staging_index)
        shutil.copyfile(args.metadata_out, staging_metadata)
        index = faiss.read_index(str(staging_index))
        resume_offset = index.ntotal
        meta_lines = _count_lines(staging_metadata)
        if meta_lines > resume_offset:
            print(
                f"Metadata has {meta_lines:,} lines but index has {resume_offset:,} "
                f"vectors. Truncating metadata to match (lost "
                f"{meta_lines - resume_offset:,} unsaved chunks; will re-embed).",
                file=sys.stderr,
                flush=True,
            )
            _truncate_jsonl(staging_metadata, resume_offset)
        elif meta_lines < resume_offset:
            raise ValueError(
                f"Index has {resume_offset} vectors but metadata only "
                f"{meta_lines} lines. Index is ahead of metadata. To recover, "
                f"copy the first {resume_offset} lines of {args.chunks} into "
                f"{args.metadata_out} before resuming."
            )
        metadata_mode = "a"
        print(
            f"Resuming from checkpoint at {resume_offset:,} chunks (staged at {staging_index})",
            file=sys.stderr,
            flush=True,
        )
    else:
        index = faiss.IndexFlatIP(DIMENSION)
        staging_index.unlink(missing_ok=True)
        staging_metadata.unlink(missing_ok=True)

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
            local_start = time.perf_counter()
            faiss.write_index(index, str(staging_index))
            metadata_file.flush()
            os.fsync(metadata_file.fileno())
            local_elapsed = time.perf_counter() - local_start
            sync_index = _sync_to_drive(staging_index, args.index_out)
            sync_meta = _sync_to_drive(staging_metadata, args.metadata_out)
            print(
                f"Checkpoint at {processed:,} chunks: local write {local_elapsed:.1f}s, "
                f"drive sync index {sync_index:.1f}s + metadata {sync_meta:.1f}s "
                f"(index.ntotal={index.ntotal:,})",
                file=sys.stderr,
                flush=True,
            )
            next_checkpoint = processed + args.checkpoint_interval

    skipped = 0
    with (
        args.chunks.open("rb") as chunks_file,
        staging_metadata.open(metadata_mode, encoding="utf-8") as metadata_file,
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

    # Final write: stage on local, then commit to drive.
    faiss.write_index(index, str(staging_index))
    sync_index = _sync_to_drive(staging_index, args.index_out)
    sync_meta = _sync_to_drive(staging_metadata, args.metadata_out)
    elapsed = time.perf_counter() - started
    new_chunks = processed - resume_offset
    print(
        f"Wrote {processed:,} vectors to {args.index_out} and metadata to "
        f"{args.metadata_out} ({new_chunks:,} new in {elapsed / 60:.1f} min; "
        f"final drive sync: index {sync_index:.1f}s + metadata {sync_meta:.1f}s)",
        file=sys.stderr,
    )
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    rc = main()
    # FlagEmbedding worker cleanup can hang after a successful long index build.
    os._exit(rc)
