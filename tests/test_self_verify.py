from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from entrypoint import parse_args
from src.llm import LLMAnswerer
from src.pipeline import answer_question
from src.schema import Prediction, Question


def _completion(content: str) -> dict[str, list[dict[str, dict[str, str]]]]:
    return {"choices": [{"message": {"content": content}}]}


def build_answerer(content: str) -> tuple[LLMAnswerer, Mock]:
    llama_cpp = ModuleType("llama_cpp")
    llama_cpp.Llama = Mock()  # type: ignore[attr-defined]
    grammar_class = Mock()
    grammar_class.from_string.side_effect = lambda grammar: grammar
    llama_cpp.LlamaGrammar = grammar_class  # type: ignore[attr-defined]
    llm = Mock()
    llm.create_chat_completion.return_value = _completion(content)

    with (
        patch.dict(sys.modules, {"llama_cpp": llama_cpp}),
        patch.object(llama_cpp, "Llama", return_value=llm),
    ):
        answerer = LLMAnswerer(model_path=Path("mock.gguf"))

    return answerer, llm


def test_verify_letter_returns_chosen_when_model_confirms() -> None:
    answerer, _ = build_answerer("B")

    answer = answerer.verify_letter("Question", ["one", "two", "three"], "B")

    assert answer == "B"


def test_verify_letter_returns_switched_when_model_changes() -> None:
    answerer, _ = build_answerer("C")

    answer = answerer.verify_letter("Question", ["one", "two", "three"], "B")

    assert answer == "C"


def test_verify_letter_falls_back_on_invalid_output() -> None:
    answerer, _ = build_answerer("Z")

    answer = answerer.verify_letter(
        "Question",
        ["one", "two", "three", "four"],
        "B",
    )

    assert answer == "B"


def test_pipeline_direct_self_verify_called_once() -> None:
    question = Question("q1", "Question", ["one", "two", "three"])
    llm = Mock()
    llm.use_cot = False
    llm.answer_mcq.return_value = "B"
    llm.verify_letter.return_value = "C"

    prediction = answer_question(question, llm=llm, use_self_verify=True)

    assert prediction == Prediction("q1", "C")
    llm.answer_mcq.assert_called_once_with(
        question.question,
        question.choices,
        context=None,
        retrieved=None,
    )
    llm.verify_letter.assert_called_once_with(
        question.question,
        question.choices,
        "B",
        context=None,
        retrieved=None,
    )
    llm.answer.assert_not_called()


def test_pipeline_cot_then_self_verify() -> None:
    question = Question("q1", "Question", ["one", "two", "three"])
    llm = Mock()
    llm.use_cot = True
    llm.answer_mcq_cot.return_value = "B"
    llm.verify_letter.return_value = "C"

    prediction = answer_question(question, llm=llm, use_self_verify=True)

    assert prediction == Prediction("q1", "C")
    llm.answer_mcq_cot.assert_called_once_with(
        question.question,
        question.choices,
        context=None,
        retrieved=None,
    )
    llm.verify_letter.assert_called_once_with(
        question.question,
        question.choices,
        "B",
        context=None,
        retrieved=None,
    )


def test_pipeline_self_verify_skips_tool_path() -> None:
    question = Question("q1", "Tính 6*7 bằng bao nhiêu?", ["41", "42"])
    llm = Mock()
    llm.verify_letter = Mock(return_value="A")
    run_python = Mock()

    with patch("src.pipeline.solve_with_tools", return_value="B") as solve:
        prediction = answer_question(
            question,
            llm=llm,
            use_tools=True,
            run_python=run_python,
            use_self_verify=True,
        )

    assert prediction == Prediction("q1", "B")
    solve.assert_called_once_with(llm, question.question, question.choices, run_python)
    llm.verify_letter.assert_not_called()


def test_self_verify_cli_uses_none_sentinel() -> None:
    assert parse_args([]).self_verify is None
    assert parse_args(["--self-verify"]).self_verify is True
    assert parse_args(["--no-self-verify"]).self_verify is False
