from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from src.llm import LLMAnswerer, _PROMPT_T1, _PROMPT_T2, _PROMPT_T3
from src.pipeline import answer_question
from src.schema import Prediction, Question


def _completion(content: str) -> dict[str, list[dict[str, dict[str, str]]]]:
    return {"choices": [{"message": {"content": content}}]}


def build_answerer(
    completions: list[str] | None = None,
    *,
    use_cot: bool = False,
    cot_max_tokens: int | None = None,
) -> tuple[LLMAnswerer, Mock]:
    llama_cpp = ModuleType("llama_cpp")
    llama_cpp.Llama = Mock()  # type: ignore[attr-defined]
    grammar_class = Mock()
    grammar_class.from_string.side_effect = lambda grammar: grammar
    llama_cpp.LlamaGrammar = grammar_class  # type: ignore[attr-defined]
    llm = Mock()
    if completions is None:
        llm.create_chat_completion.return_value = _completion("A")
    else:
        llm.create_chat_completion.side_effect = [
            _completion(content) for content in completions
        ]

    kwargs = {"model_path": Path("mock.gguf"), "use_cot": use_cot}
    if cot_max_tokens is not None:
        kwargs["cot_max_tokens"] = cot_max_tokens

    with (
        patch.dict(sys.modules, {"llama_cpp": llama_cpp}),
        patch.object(llama_cpp, "Llama", return_value=llm),
    ):
        answerer = LLMAnswerer(**kwargs)

    return answerer, llm


def test_cot_v10_uses_4_step_reasoning() -> None:
    answerer, llm = build_answerer(["Suy luận. Đáp án cuối: B", "B"])

    answerer.answer_mcq_cot("Question", ["one", "two", "three", "four"])

    system_prompt = llm.create_chat_completion.call_args_list[0].kwargs["messages"][0][
        "content"
    ]
    for step in ("Bước 1", "Bước 2", "Bước 3", "Bước 4"):
        assert step in system_prompt


def test_cot_v10_default_max_tokens_350() -> None:
    answerer, _ = build_answerer(use_cot=True)

    assert answerer.cot_max_tokens == 350


def test_cot_v10_self_consistent_calls_3_extractions() -> None:
    answerer, llm = build_answerer(["Reasoning", "A", "A", "B"])

    answer = answerer.answer_mcq_cot_self_consistent(
        "Question",
        ["one", "two", "three"],
    )

    assert answer == "A"
    assert llm.create_chat_completion.call_count == 4
    first_call, *extraction_calls = llm.create_chat_completion.call_args_list
    assert "grammar" not in first_call.kwargs
    for call, instruction in zip(
        extraction_calls,
        (_PROMPT_T1, _PROMPT_T2, _PROMPT_T3),
    ):
        assert "grammar" in call.kwargs
        assert call.kwargs["max_tokens"] == 2
        assert "Reasoning" in call.kwargs["messages"][1]["content"]
        assert instruction in call.kwargs["messages"][1]["content"]


def test_cot_v10_self_consistent_tiebreak_t1() -> None:
    answerer, _ = build_answerer(["Reasoning", "A", "B", "C"])

    answer = answerer.answer_mcq_cot_self_consistent(
        "Question",
        ["one", "two", "three"],
    )

    assert answer == "A"


def test_pipeline_routes_to_cot_when_use_cot_true() -> None:
    answerer, llm = build_answerer(["Reasoning", "B"], use_cot=True)
    question = Question("q1", "Question", ["one", "two"])

    prediction = answer_question(question, llm=answerer)

    assert prediction == Prediction("q1", "B")
    assert llm.create_chat_completion.call_count == 2
    first_call, second_call = llm.create_chat_completion.call_args_list
    assert "grammar" not in first_call.kwargs
    assert "grammar" in second_call.kwargs


def test_pipeline_routes_to_cot_sc_when_both_true() -> None:
    answerer, llm = build_answerer(["Reasoning", "C", "C", "A"], use_cot=True)
    question = Question("q1", "Question", ["one", "two", "three"])

    prediction = answer_question(
        question,
        llm=answerer,
        self_consistency=True,
    )

    assert prediction == Prediction("q1", "C")
    assert llm.create_chat_completion.call_count == 4
    first_call, *extraction_calls = llm.create_chat_completion.call_args_list
    assert "grammar" not in first_call.kwargs
    assert all("grammar" in call.kwargs for call in extraction_calls)


def test_pipeline_cot_skipped_on_tool_path() -> None:
    answerer, _ = build_answerer(use_cot=True)
    question = Question("q1", "Tính 6*7 bằng bao nhiêu?", ["41", "42"])
    run_python = Mock()

    with (
        patch("src.pipeline.solve_with_tools", return_value="B") as solve,
        patch.object(answerer, "answer_mcq_cot", return_value="A") as cot,
        patch.object(
            answerer,
            "answer_mcq_cot_self_consistent",
            return_value="A",
        ) as cot_sc,
    ):
        prediction = answer_question(
            question,
            llm=answerer,
            use_tools=True,
            run_python=run_python,
        )

    assert prediction == Prediction("q1", "B")
    solve.assert_called_once_with(answerer, question.question, question.choices, run_python)
    cot.assert_not_called()
    cot_sc.assert_not_called()
