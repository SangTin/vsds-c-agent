import csv
from pathlib import Path

from src.schema import Prediction


def discover_input(data_dir: Path) -> Path:
    candidates: list[tuple[int, int, str, Path]] = []
    prefixes = {
        "private": ("private_test", "private-test"),
        "public": ("public_test", "public-test"),
    }

    if data_dir.is_dir():
        for path in data_dir.iterdir():
            suffix = path.suffix.lower()
            if not path.is_file() or suffix not in {".json", ".csv"}:
                continue
            stem = path.stem.lower()
            for privacy_priority, kind in enumerate(("private", "public")):
                matches_kind = any(
                    stem == prefix or stem.startswith(f"{prefix}_")
                    for prefix in prefixes[kind]
                )
                if matches_kind:
                    format_priority = 0 if suffix == ".json" else 1
                    candidates.append(
                        (privacy_priority, format_priority, path.name.lower(), path)
                    )
                    break

    if not candidates:
        raise FileNotFoundError(
            f"No supported private/public test JSON or CSV file found in {data_dir}"
        )
    return min(candidates)[-1]


def write_predictions(preds: list[Prediction], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["qid", "answer"])
        writer.writerows((pred.qid, pred.answer) for pred in preds)
