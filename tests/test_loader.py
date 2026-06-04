import csv
import json
from pathlib import Path

import pytest

from src.loader import load_questions


def test_load_json_preserves_order_and_choices(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    data = [
        {"qid": "q1", "question": "Four?", "choices": ["A", "B", "C", "D"]},
        {"qid": "q2", "question": "Ten?", "choices": [str(i) for i in range(10)]},
        {"qid": "q3", "question": "Two?", "choices": ["Yes", "No"]},
    ]
    path.write_text(json.dumps(data), encoding="utf-8")

    questions = load_questions(path)

    assert [question.qid for question in questions] == ["q1", "q2", "q3"]
    assert [len(question.choices) for question in questions] == [4, 10, 2]
    assert questions[1].question == "Ten?"


def test_load_csv(tmp_path: Path) -> None:
    path = tmp_path / "questions.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["qid", "question", "choices"])
        writer.writerow(["q1", "First?", json.dumps(["One", "Two"])])
        writer.writerow(["q2", "Second?", json.dumps(["A", "B", "C"])])

    questions = load_questions(path)

    assert [question.qid for question in questions] == ["q1", "q2"]
    assert questions[1].choices == ["A", "B", "C"]


def test_duplicate_qid_fails_with_offending_qid(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        json.dumps(
            [
                {"qid": "same", "question": "One?", "choices": ["A"]},
                {"qid": "same", "question": "Two?", "choices": ["B"]},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="same"):
        load_questions(path)


def test_empty_choices_fails_with_offending_qid(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(
        json.dumps([{"qid": "empty-qid", "question": "None?", "choices": []}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="empty-qid"):
        load_questions(path)


def test_unsupported_extension_fails(tmp_path: Path) -> None:
    path = tmp_path / "questions.txt"
    path.write_text("not supported", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported input extension"):
        load_questions(path)
