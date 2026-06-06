from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from entrypoint import load_config, parse_args
from src.pipeline import answer_question
from src.schema import Prediction, Question


PASSAGE_MARKER = "\u0110o\u1ea1n th\u00f4ng tin:"


def _passage_question(text: str = "N\u1ed9i dung c\u00f3 s\u1eb5n.") -> Question:
    return Question(
        "q1",
        f"{PASSAGE_MARKER}\n{text}\nC\u00e2u h\u1ecfi: Ch\u1ecdn \u0111\u00e1p \u00e1n?",
        ["one", "two"],
    )


def _cot_llm(answer: str = "B") -> Mock:
    llm = Mock()
    llm.use_cot = True
    llm.answer_mcq.return_value = answer
    llm.answer_mcq_cot.return_value = answer
    llm.answer_mcq_self_consistent.return_value = answer
    llm.answer_mcq_cot_self_consistent.return_value = answer
    return llm


def test_pipeline_passage_default_skips_cot() -> None:
    question = _passage_question()
    llm = _cot_llm("B")

    prediction = answer_question(question, llm=llm, use_cot_passage=False)

    assert prediction == Prediction("q1", "B")
    llm.answer_mcq.assert_called_once_with(
        question.question,
        question.choices,
        context=None,
        retrieved=None,
    )
    llm.answer_mcq_cot.assert_not_called()


def test_pipeline_passage_enabled_uses_cot() -> None:
    question = _passage_question()
    llm = _cot_llm("B")

    prediction = answer_question(question, llm=llm, use_cot_passage=True)

    assert prediction == Prediction("q1", "B")
    llm.answer_mcq_cot.assert_called_once_with(
        question.question,
        question.choices,
        context=None,
        retrieved=None,
    )
    llm.answer_mcq.assert_not_called()


def test_pipeline_passage_oversize_falls_back(capsys) -> None:
    question = _passage_question("x" * 5000)
    llm = _cot_llm("B")

    prediction = answer_question(
        question,
        llm=llm,
        use_cot_passage=True,
        cot_passage_max_chars=3500,
    )

    assert prediction == Prediction("q1", "B")
    llm.answer_mcq.assert_called_once_with(
        question.question,
        question.choices,
        context=None,
        retrieved=None,
    )
    llm.answer_mcq_cot.assert_not_called()
    assert "skipping CoT for passage question 'q1'" in capsys.readouterr().err


def test_pipeline_non_passage_unaffected() -> None:
    for use_cot_passage in (False, True):
        question = Question("q1", "Question", ["one", "two"])
        llm = _cot_llm("B")

        prediction = answer_question(
            question,
            llm=llm,
            use_cot_passage=use_cot_passage,
        )

        assert prediction == Prediction("q1", "B")
        llm.answer_mcq_cot.assert_called_once_with(
            question.question,
            question.choices,
            context=None,
            retrieved=None,
        )
        llm.answer_mcq.assert_not_called()


def test_cot_passage_cli_uses_none_sentinel() -> None:
    assert parse_args([]).cot_passage is None
    assert parse_args(["--cot-passage"]).cot_passage is True
    assert parse_args(["--no-cot-passage"]).cot_passage is False
    assert parse_args(["--cot-passage-max-chars", "2000"]).cot_passage_max_chars == 2000


def test_load_config_reads_cot_passage_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    yaml = ModuleType("yaml")
    yaml.safe_load = Mock(  # type: ignore[attr-defined]
        return_value={
            "model": {
                "use_cot_passage": True,
                "cot_passage_max_chars": 2000,
            }
        }
    )

    with patch.dict(sys.modules, {"yaml": yaml}):
        model_config = load_config(config_path)["model"]

    assert model_config["use_cot_passage"] is True
    assert model_config["cot_passage_max_chars"] == 2000
