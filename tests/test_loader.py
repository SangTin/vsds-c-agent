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


def test_json_tolerates_extra_keys(tmp_path: Path) -> None:
    path = tmp_path / "extra.json"
    path.write_text(
        json.dumps(
            [{"qid": "q1", "question": "X?", "choices": ["A", "B"], "category": "z", "id": 7}]
        ),
        encoding="utf-8",
    )
    questions = load_questions(path)
    assert questions[0].choices == ["A", "B"]


def test_csv_letter_columns_layout(tmp_path: Path) -> None:
    path = tmp_path / "letters.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "qid", "question", "A", "B", "C", "D"])
        writer.writerow(["0", "q1", "Pick?", "Hà Nội", "Huế", "Đà Nẵng", "TP.HCM"])
        writer.writerow(["1", "q2", "Two?", "Yes", "No", "", ""])

    questions = load_questions(path)
    assert questions[0].choices == ["Hà Nội", "Huế", "Đà Nẵng", "TP.HCM"]
    # Trailing empty letter columns are trimmed.
    assert questions[1].choices == ["Yes", "No"]


def test_csv_with_bom_and_extra_column(tmp_path: Path) -> None:
    path = tmp_path / "bom.csv"
    # Leading BOM + an extra 'category' column (Excel-style export).
    content = "﻿qid,question,choices,category\n"
    content += 'q1,"First?","[""One"", ""Two""]",x\n'
    path.write_text(content, encoding="utf-8")

    questions = load_questions(path)
    assert questions[0].qid == "q1"
    assert questions[0].choices == ["One", "Two"]
