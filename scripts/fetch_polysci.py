"""Download the open-access Marxist-Leninist textbook PDFs we can fetch directly.

Covers 2 of the 5 mandatory Vietnamese university political-theory subjects:
    - Triết học Mác-Lênin
    - Kinh tế Chính trị Mác-Lênin
hosted by the Marxism-Leninism faculty of Duy Tan University.

The other 3 subjects (Tư tưởng Hồ Chí Minh, Chủ nghĩa Xã hội Khoa học, Lịch sử
Đảng CSVN) do not have a clean public direct-PDF URL we can rely on; drop those
PDFs into the same raw directory manually. The downstream chunker
(scripts/build_targeted_corpus.py) parses every PDF in the folder regardless of
source.

Usage:
    python scripts/fetch_polysci.py [--raw data_kb/polysci/raw]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

SOURCES = {
    "triet-mac-lenin.pdf": (
        "https://kmacle.duytan.edu.vn/uploads/"
        "75770b9b-cdbf-4038-90e2-f25e1f4426fe_triethocmaclenin.pdf"
    ),
    "kinh-te-chinh-tri-mac-lenin.pdf": (
        "https://kmacle.duytan.edu.vn/uploads/"
        "2eae0e86-e7c6-431d-aa08-b8f89ba71921_giaotrinhkinhtechinhtri.pdf"
    ),
}

USER_AGENT = "Mozilla/5.0 (vsds-bangc-agent fetcher)"


def download(url: str, dest: Path) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as out:
        n = 0
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
            n += len(chunk)
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data_kb/polysci/raw"))
    args = parser.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)

    for name, url in SOURCES.items():
        dest = args.raw / name
        if dest.exists() and dest.stat().st_size > 100_000:
            print(f"  skip {name} (already present, {dest.stat().st_size:,} bytes)", file=sys.stderr)
            continue
        print(f"  fetch {name} ...", file=sys.stderr)
        try:
            size = download(url, dest)
            print(f"    -> {size:,} bytes", file=sys.stderr)
        except Exception as exc:
            print(f"    FAILED {type(exc).__name__}: {exc}", file=sys.stderr)
            dest.unlink(missing_ok=True)

    print(file=sys.stderr)
    print("Auto-fetched 2/5 mandatory subjects. Please drop the remaining 3 PDFs into", file=sys.stderr)
    print(f"  {args.raw}/", file=sys.stderr)
    print("with descriptive filenames (used as titles), e.g.:", file=sys.stderr)
    print("  tu-tuong-ho-chi-minh.pdf", file=sys.stderr)
    print("  chu-nghia-xa-hoi-khoa-hoc.pdf", file=sys.stderr)
    print("  lich-su-dang-cong-san-vn.pdf", file=sys.stderr)
    print("then run: python scripts/build_targeted_corpus.py --raw", args.raw,
          "--out data_kb/polysci/chunks.jsonl", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
