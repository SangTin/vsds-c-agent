import csv
import json
from pathlib import Path
from typing import Any

from src.schema import Question


def _build_question(item: Any, source: str) -> Question:
    if not isinstance(item, dict):
        raise ValueError(f"Invalid question <unknown> in {source}: expected an object")

    qid = item.get("qid", "<unknown>")
    required_keys = {"qid", "question", "choices"}
    if set(item) != required_keys:
        raise ValueError(
            f"Invalid question {qid!r} in {source}: "
            f"expected keys {sorted(required_keys)}"
        )
    if not isinstance(item["qid"], str) or not isinstance(item["question"], str):
        raise ValueError(
            f"Invalid question {qid!r} in {source}: qid and question must be strings"
        )

    try:
        return Question(
            qid=item["qid"],
            question=item["question"],
            choices=item["choices"],
        )
    except ValueError as exc:
        raise ValueError(f"Invalid question {qid!r} in {source}: {exc}") from exc


def _load_json(path: Path) -> list[Question]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Invalid JSON in {path}: expected a list of questions")
    return [_build_question(item, str(path)) for item in data]


def _load_csv(path: Path) -> list[Question]:
    questions: list[Question] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required_columns = {"qid", "question", "choices"}
        if reader.fieldnames is None or set(reader.fieldnames) != required_columns:
            raise ValueError(
                f"Invalid CSV in {path}: expected columns {sorted(required_columns)}"
            )

        for row in reader:
            qid = row.get("qid") or "<unknown>"
            try:
                choices = json.loads(row["choices"])
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(
                    f"Invalid question {qid!r} in {path}: choices must be JSON"
                ) from exc
            questions.append(
                _build_question(
                    {
                        "qid": row["qid"],
                        "question": row["question"],
                        "choices": choices,
                    },
                    str(path),
                )
            )
    return questions


def load_questions(path: Path) -> list[Question]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        questions = _load_json(path)
    elif suffix == ".csv":
        questions = _load_csv(path)
    else:
        raise ValueError(f"Unsupported input extension {path.suffix!r} for {path}")

    seen: set[str] = set()
    for question in questions:
        if question.qid in seen:
            raise ValueError(f"Duplicate qid {question.qid!r} in {path}")
        seen.add(question.qid)
    return questions
