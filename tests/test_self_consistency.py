from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from entrypoint import load_config, parse_args
from src.llm import LLMAnswerer
from src.pipeline import answer_question
from src.schema import Prediction, Question


def test_self_consistent_majority_2of3() -> None:
    answerer = object.__new__(LLMAnswerer)

    with patch.object(answerer, "answer_mcq", side_effect=["A", "A", "B"]):
        answer = answerer.answer_mcq_self_consistent("Question", ["one", "two"])

    assert answer == "A"


def test_self_consistent_3way_tie_uses_t1() -> None:
    answerer = object.__new__(LLMAnswerer)

    with patch.object(answerer, "answer_mcq", side_effect=["A", "B", "C"]):
        answer = answerer.answer_mcq_self_consistent(
            "Question",
            ["one", "two", "three"],
        )

    assert answer == "A"


def test_self_consistent_unanimous() -> None:
    answerer = object.__new__(LLMAnswerer)

    with patch.object(answerer, "answer_mcq", side_effect=["D", "D", "D"]):
        answer = answerer.answer_mcq_self_consistent(
            "Question",
            ["one", "two", "three", "four"],
        )

    assert answer == "D"


def test_pipeline_self_consistency_direct_path() -> None:
    question = Question("q1", "Question", ["one", "two"])
    llm = Mock()
    llm.answer_mcq_self_consistent.return_value = "B"

    prediction = answer_question(question, llm=llm, self_consistency=True)

    assert prediction == Prediction("q1", "B")
    llm.answer_mcq_self_consistent.assert_called_once_with(
        question.question,
        question.choices,
        context=None,
        retrieved=None,
    )
    llm.answer.assert_not_called()


def test_pipeline_self_consistency_skips_tool_path() -> None:
    question = Question("q1", "Tính 6*7 bằng bao nhiêu?", ["41", "42"])
    llm = Mock()
    llm.answer_mcq_self_consistent.return_value = "A"
    run_python = Mock()

    with patch("src.pipeline.solve_with_tools", return_value="B") as solve:
        prediction = answer_question(
            question,
            llm=llm,
            use_tools=True,
            run_python=run_python,
            self_consistency=True,
        )

    assert prediction == Prediction("q1", "B")
    solve.assert_called_once_with(llm, question.question, question.choices, run_python)
    llm.answer_mcq_self_consistent.assert_not_called()


def test_pipeline_self_consistency_with_legal_rag() -> None:
    question = Question(
        "q1",
        "Theo Bộ luật Hình sự, độ tuổi chịu trách nhiệm hình sự là bao nhiêu?",
        ["14 tuổi", "16 tuổi"],
    )
    llm = Mock()
    llm.answer_mcq_self_consistent.return_value = "B"
    retriever = Mock()
    retriever.retrieve.return_value = [
        {
            "doc_id": "blhs",
            "title": "Bộ luật Hình sự 2015",
            "text": "context",
            "score": 0.9,
        }
    ]

    prediction = answer_question(
        question,
        llm=llm,
        legal_retriever=retriever,
        legal_min_score=0.7,
        self_consistency=True,
    )

    assert prediction == Prediction("q1", "B")
    context = llm.answer_mcq_self_consistent.call_args.kwargs["context"]
    assert "Trích văn bản pháp luật liên quan:" in context
    llm.answer_mcq.assert_not_called()


def test_self_consistency_cli_uses_none_sentinel() -> None:
    assert parse_args([]).self_consistency is None
    assert parse_args(["--self-consistency"]).self_consistency is True
    assert parse_args(["--no-self-consistency"]).self_consistency is False


def test_load_config_reads_self_consistency_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    yaml = ModuleType("yaml")
    yaml.safe_load = Mock(  # type: ignore[attr-defined]
        return_value={"self_consistency": {"enabled": True}}
    )

    with patch.dict(sys.modules, {"yaml": yaml}):
        config = load_config(config_path)

    assert config["self_consistency"] == {"enabled": True}
