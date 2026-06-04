import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

from entrypoint import load_config, parse_args
from src.llm import LLMAnswerer
from src.pipeline import answer_question
from src.schema import Prediction, Question


def build_answerer(
    content: str = "C",
    *,
    use_cot: bool = False,
    cot_max_tokens: int = 200,
) -> tuple[LLMAnswerer, Mock, Mock]:
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
        answerer = LLMAnswerer(
            model_path=Path("mock.gguf"),
            use_cot=use_cot,
            cot_max_tokens=cot_max_tokens,
        )

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


def test_four_choice_prompt_does_not_contain_many_choice_hint() -> None:
    answerer, llm, _ = build_answerer("B")

    answerer.answer_mcq("Question", ["one", "two", "three", "four"])

    prompt = llm.create_chat_completion.call_args.kwargs["messages"][1]["content"]
    assert "Lưu ý: có" not in prompt


def test_ten_choice_prompt_contains_many_choice_hint() -> None:
    answerer, llm, _ = build_answerer("J")

    answerer.answer_mcq("Question", [str(index) for index in range(10)])

    prompt = llm.create_chat_completion.call_args.kwargs["messages"][1]["content"]
    assert "có 10 lựa chọn từ A đến J" in prompt


def test_answer_mcq_returns_mocked_completion_letter() -> None:
    answerer, _, _ = build_answerer("C")

    assert answerer.answer_mcq("Question", ["one", "two", "three"]) == "C"


def test_answer_mcq_raises_on_empty_completion_content() -> None:
    answerer, _, _ = build_answerer("")

    with pytest.raises(ValueError, match="did not contain an answer"):
        answerer.answer_mcq("Question", ["one", "two"])


def test_answer_mcq_cot_uses_two_calls_and_returns_extracted_letter() -> None:
    answerer, llm, _ = build_answerer(cot_max_tokens=123)
    llm.create_chat_completion.side_effect = [
        {"choices": [{"message": {"content": "Suy luận ngắn. Đáp án cuối: B"}}]},
        {"choices": [{"message": {"content": "B"}}]},
    ]

    answer = answerer.answer_mcq_cot("Question", [str(index) for index in range(10)])

    assert answer == "B"
    assert llm.create_chat_completion.call_count == 2
    first_call, second_call = llm.create_chat_completion.call_args_list
    assert "grammar" not in first_call.kwargs
    assert first_call.kwargs["max_tokens"] == 123
    assert "grammar" in second_call.kwargs
    assert second_call.kwargs["max_tokens"] == 2
    assert (
        first_call.kwargs["messages"][1]["content"].count("Lưu ý: có 10 lựa chọn")
        == 1
    )
    assert (
        second_call.kwargs["messages"][1]["content"].count("Lưu ý: có 10 lựa chọn")
        == 1
    )


def test_answer_mcq_cot_still_extracts_after_empty_reasoning() -> None:
    answerer, llm, _ = build_answerer()
    llm.create_chat_completion.side_effect = [
        {"choices": [{"message": {"content": ""}}]},
        {"choices": [{"message": {"content": "A"}}]},
    ]

    assert answerer.answer_mcq_cot("Question", ["one", "two"]) == "A"
    assert llm.create_chat_completion.call_count == 2


@pytest.mark.parametrize(
    ("use_cot", "expected_method"),
    [(False, "answer_mcq"), (True, "answer_mcq_cot")],
)
def test_answer_dispatches_by_use_cot(use_cot: bool, expected_method: str) -> None:
    answerer, _, _ = build_answerer(use_cot=use_cot)

    with (
        patch.object(answerer, "answer_mcq", return_value="A") as fast,
        patch.object(answerer, "answer_mcq_cot", return_value="B") as cot,
    ):
        answer = answerer.answer("Question", ["one", "two"])

    assert answer == ("B" if use_cot else "A")
    expected = cot if expected_method == "answer_mcq_cot" else fast
    unexpected = fast if expected_method == "answer_mcq_cot" else cot
    expected.assert_called_once_with("Question", ["one", "two"], None)
    unexpected.assert_not_called()


def test_cot_cli_uses_none_sentinel_and_supports_explicit_false() -> None:
    assert parse_args([]).cot is None
    assert parse_args(["--cot"]).cot is True
    assert parse_args(["--no-cot"]).cot is False


def test_load_config_reads_cot_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    yaml = ModuleType("yaml")
    yaml.safe_load = Mock(  # type: ignore[attr-defined]
        return_value={"model": {"use_cot": True, "cot_max_tokens": 321}}
    )

    with patch.dict(sys.modules, {"yaml": yaml}):
        model_config = load_config(config_path)["model"]

    assert model_config["use_cot"] is True
    assert model_config["cot_max_tokens"] == 321


def test_pipeline_without_llm_returns_stub_fallback() -> None:
    question = Question("q1", "Question", ["one", "two"])

    assert answer_question(question, fallback="B", llm=None) == Prediction("q1", "B")


def test_pipeline_returns_llm_answer() -> None:
    question = Question("q1", "Question", ["one", "two"])
    llm = Mock()
    llm.answer.return_value = "B"

    assert answer_question(question, llm=llm) == Prediction("q1", "B")


def test_pipeline_falls_back_when_llm_raises() -> None:
    question = Question("q1", "Question", ["one", "two"])
    llm = Mock()
    llm.answer.side_effect = RuntimeError("inference failed")

    assert answer_question(question, llm=llm) == Prediction("q1", "A")
