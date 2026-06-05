"""Ingest a folder of source documents into chunked JSONL for targeted RAG.

Unlike build_viwiki_corpus.py (which streams Wikipedia), this reads local files
you drop into a raw directory — Vietnamese legal texts, HCM/Marx-Lenin textbooks,
etc. — and emits the same schema the FAISS index builder expects:
    {"doc_id": str, "title": str, "chunk_id": int, "text": str}

Supported inputs (by extension): .txt, .md, .json/.jsonl (already-chunked or
list-of-strings), .html/.htm, .pdf. PDF needs `pypdf`, HTML needs `beautifulsoup4`
(both lazy-imported; install via requirements-data.txt).

Usage:
    python scripts/build_targeted_corpus.py \
        --raw data_kb/targeted/raw \
        --out data_kb/targeted/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CHUNK_TARGET_CHARS = 1500
CHUNK_MIN_CHARS = 120
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    return _SLUG_RE.sub("-", name.lower()).strip("-") or "doc"


def split_text(text: str, target: int = CHUNK_TARGET_CHARS) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for para in paragraphs:
        # A single huge paragraph (e.g. a long article) is hard-split on target.
        while len(para) > target * 1.5:
            head, para = para[:target], para[target:]
            chunks.append(head.strip())
        if buf_len + len(para) > target and buf:
            chunks.append("\n\n".join(buf))
            buf, buf_len = [para], len(para)
        else:
            buf.append(para)
            buf_len += len(para) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    return [c for c in chunks if len(c) >= CHUNK_MIN_CHARS]


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text("\n")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix in {".json", ".jsonl"}:
        # Either a JSON list of strings, or JSONL — joined into one text blob.
        raw = path.read_text(encoding="utf-8", errors="ignore").strip()
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return "\n\n".join(str(x) for x in data)
            return str(data)
        except json.JSONDecodeError:
            return "\n\n".join(line for line in raw.splitlines() if line.strip())
    raise ValueError(f"Unsupported file type: {path.name}")


def iter_documents(raw_dir: Path):
    for path in sorted(raw_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {
            ".txt", ".md", ".html", ".htm", ".pdf", ".json", ".jsonl"
        }:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data_kb/targeted/raw"))
    parser.add_argument("--out", type=Path, default=Path("data_kb/targeted/chunks.jsonl"))
    args = parser.parse_args()

    if not args.raw.is_dir():
        print(f"raw dir not found: {args.raw}", file=sys.stderr)
        print("Drop source documents (.txt/.md/.pdf/.html/.json) there first.", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    docs = 0
    chunks_written = 0
    with args.out.open("w", encoding="utf-8") as f:
        for path in iter_documents(args.raw):
            try:
                text = extract_text(path)
            except Exception as exc:  # report and skip a bad file, keep going
                print(f"skip {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            title = path.stem
            doc_id = slugify(path.relative_to(args.raw).as_posix().rsplit(".", 1)[0])
            chunks = split_text(text)
            for chunk_id, chunk in enumerate(chunks):
                f.write(json.dumps(
                    {"doc_id": doc_id, "title": title, "chunk_id": chunk_id, "text": chunk},
                    ensure_ascii=False,
                ))
                f.write("\n")
            docs += 1
            chunks_written += len(chunks)
            print(f"  {path.name}: {len(chunks)} chunks", file=sys.stderr)

    print(f"done: {docs} documents -> {chunks_written} chunks at {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
