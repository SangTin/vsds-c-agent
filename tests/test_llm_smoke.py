import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

from src.llm import LLMAnswerer
from src.pipeline import answer_question
from src.schema import Prediction, Question


def build_answerer(content: str = "C") -> tuple[LLMAnswerer, Mock, Mock]:
    llama_cpp = ModuleType("llama_cpp")
    llama_cpp.Llama = Mock()  # type: ignore[attr-defined]
    grammar_class = Mock()
    grammar_class.from_string.side_effect = lambda grammar: grammar
    llama_cpp.LlamaGrammar = grammar_class  # type: ignore[attr-defined]
    llm = Mock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": content}}]
    }

    with (
        patch.dict(sys.modules, {"llama_cpp": llama_cpp}),
        patch.object(llama_cpp, "Llama", return_value=llm),
    ):
        answerer = LLMAnswerer(model_path=Path("mock.gguf"))

    return answerer, llm, grammar_class.from_string


@pytest.mark.parametrize("n_choices", [2, 4, 10])
def test_grammar_contains_each_expected_letter(n_choices: int) -> None:
    answerer, _, grammar_from_string = build_answerer()

    grammar = answerer.grammar_for(n_choices)

    assert grammar == "root ::= " + " | ".join(
        f'"{letter}"' for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_choices]
    )
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_choices]:
        assert f'"{letter}"' in grammar
    assert grammar_from_string.call_count == 1
    assert answerer.grammar_for(n_choices) is grammar
    assert grammar_from_string.call_count == 1


def test_prompt_contains_question_and_letter_prefixed_choices() -> None:
    answerer, llm, _ = build_answerer("B")

    answerer.answer_mcq("Câu hỏi thử?", ["Lựa chọn một", "Lựa chọn hai"])

    messages = llm.create_chat_completion.call_args.kwargs["messages"]
    prompt = messages[1]["content"]
    assert "Câu hỏi thử?" in prompt
    assert "A. Lựa chọn một" in prompt
    assert "B. Lựa chọn hai" in prompt


def test_answer_mcq_returns_mocked_completion_letter() -> None:
    answerer, _, _ = build_answerer("C")

    assert answerer.answer_mcq("Question", ["one", "two", "three"]) == "C"


def test_answer_mcq_raises_on_empty_completion_content() -> None:
    answerer, _, _ = build_answerer("")

    with pytest.raises(ValueError, match="did not contain an answer"):
        answerer.answer_mcq("Question", ["one", "two"])


def test_pipeline_without_llm_returns_stub_fallback() -> None:
    question = Question("q1", "Question", ["one", "two"])

    assert answer_question(question, fallback="B", llm=None) == Prediction("q1", "B")


def test_pipeline_returns_llm_answer() -> None:
    question = Question("q1", "Question", ["one", "two"])
    llm = Mock()
    llm.answer_mcq.return_value = "B"

    assert answer_question(question, llm=llm) == Prediction("q1", "B")


def test_pipeline_falls_back_when_llm_raises() -> None:
    question = Question("q1", "Question", ["one", "two"])
    llm = Mock()
    llm.answer_mcq.side_effect = RuntimeError("inference failed")

    assert answer_question(question, llm=llm) == Prediction("q1", "A")
