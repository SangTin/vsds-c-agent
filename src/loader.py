import csv
import json
import string
from pathlib import Path
from typing import Any

from src.schema import Question

# Letter columns for the alternative CSV layout (qid,question,A,B,C,D,...).
_LETTER_COLUMNS = list(string.ascii_uppercase)  # A..Z


def _build_question(qid: Any, question: Any, choices: Any, source: str) -> Question:
    label = qid if isinstance(qid, str) and qid else "<unknown>"
    if not isinstance(qid, str) or not isinstance(question, str):
        raise ValueError(
            f"Invalid question {label!r} in {source}: qid and question must be strings"
        )
    try:
        return Question(qid=qid, question=question, choices=choices)
    except ValueError as exc:
        raise ValueError(f"Invalid question {label!r} in {source}: {exc}") from exc


def _load_json(path: Path) -> list[Question]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Invalid JSON in {path}: expected a list of questions")
    questions: list[Question] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(
                f"Invalid item in {path}: expected an object, got {type(item).__name__}"
            )
        # Tolerate extra keys (BTC may add id/category/etc.); require only the three we use.
        missing = {"qid", "question", "choices"} - set(item)
        if missing:
            raise ValueError(
                f"Invalid question {item.get('qid', '<unknown>')!r} in {path}: "
                f"missing keys {sorted(missing)}"
            )
        questions.append(
            _build_question(item["qid"], item["question"], item["choices"], str(path))
        )
    return questions


def _parse_choices_cell(raw: Any) -> list[str]:
    """Parse a `choices` cell: a JSON list string, or a newline/pipe/semicolon
    delimited list as a fallback for non-JSON CSV exports."""
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str):
        raise ValueError("choices cell must be a JSON list or delimited string")
    text = raw.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    for sep in ("\n", "|", ";"):
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]
    raise ValueError("choices cell is not a JSON list or a recognizable delimited list")


def _choices_from_letter_columns(row: dict[str, Any]) -> list[str]:
    """Assemble choices from contiguous A,B,C,... columns (qid,question,A,B,C,D layout)."""
    choices: list[str] = []
    for letter in _LETTER_COLUMNS:
        if letter not in row:
            break
        value = row[letter]
        if value is None or str(value).strip() == "":
            break
        choices.append(str(value).strip())
    return choices


def _load_csv(path: Path) -> list[Question]:
    questions: list[Question] = []
    # utf-8-sig strips a leading BOM if BTC exports the CSV from Excel.
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        if "qid" not in fields or "question" not in fields:
            raise ValueError(
                f"Invalid CSV in {path}: must have at least 'qid' and 'question' columns "
                f"(got {sorted(fields)})"
            )
        has_choices_col = "choices" in fields
        has_letter_cols = "A" in fields and "B" in fields
        if not has_choices_col and not has_letter_cols:
            raise ValueError(
                f"Invalid CSV in {path}: need a 'choices' column or letter columns A,B,C,..."
            )
        for row in reader:
            qid = (row.get("qid") or "<unknown>").strip()
            if has_choices_col and row.get("choices") not in (None, ""):
                try:
                    choices = _parse_choices_cell(row["choices"])
                except ValueError as exc:
                    raise ValueError(f"Invalid question {qid!r} in {path}: {exc}") from exc
            else:
                choices = _choices_from_letter_columns(row)
            questions.append(
                _build_question(row.get("qid"), row.get("question"), choices, str(path))
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
