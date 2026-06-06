from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

from entrypoint import parse_args
from src.agent import loop
from src.agent.loop import solve_with_tools_ranking
from src.pipeline import answer_question
from src.schema import Prediction, Question


class MockRankingLLM:
    seed = 42

    def __init__(self, completions: list[str]) -> None:
        self.completions = completions
        self.complete_calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.complete_calls.append({"messages": messages, "kwargs": kwargs})
        if len(self.complete_calls) <= len(self.completions):
            return self.completions[len(self.complete_calls) - 1]
        return self.completions[-1]


def test_ranking_instruction_constant_exists() -> None:
    assert "thay từng giá trị" in loop._RANKING_INSTRUCTION
    assert "in ra chữ cái" in loop._RANKING_INSTRUCTION


def test_solve_ranking_returns_letter_on_valid_output() -> None:
    llm = MockRankingLLM(["```python\nprint('B')\n```"])
    run_python = Mock(return_value={"stdout": "B"})

    answer = solve_with_tools_ranking(
        llm,
        "Tính Q tại từng đáp án.",
        ["A choice", "B choice", "C choice"],
        run_python=run_python,
    )

    assert answer == "B"
    run_python.assert_called_once_with("print('B')")


def test_solve_ranking_validates_letter_against_choices() -> None:
    llm = MockRankingLLM(["```python\nprint('Z')\n```"])
    run_python = Mock(return_value={"stdout": "Z"})

    answer = solve_with_tools_ranking(
        llm,
        "Tính Q tại từng đáp án.",
        ["A", "B", "C", "D"],
        run_python=run_python,
        max_retries=0,
    )

    assert answer is None


def test_solve_ranking_retries_on_error() -> None:
    llm = MockRankingLLM(
        [
            "```python\nraise RuntimeError('boom')\n```",
            "```python\nprint('C')\n```",
        ]
    )
    run_python = Mock(side_effect=[RuntimeError("boom"), {"stdout": "C"}])

    answer = solve_with_tools_ranking(
        llm,
        "Tính Q tại từng đáp án.",
        ["A", "B", "C"],
        run_python=run_python,
    )

    assert answer == "C"
    assert run_python.call_count == 2
    assert len(llm.complete_calls) == 2
    retry_prompt = llm.complete_calls[1]["messages"][-1]["content"]
    assert "boom" in retry_prompt


def test_pipeline_pot_ranking_branch() -> None:
    question = Question("q1", "Tính 6*7 bằng bao nhiêu?", ["41", "42"])
    llm = Mock()
    run_python = Mock()

    with (
        patch("src.pipeline.solve_with_tools_ranking", return_value="B") as ranking,
        patch("src.pipeline.solve_with_tools", return_value="A") as default_pot,
    ):
        prediction = answer_question(
            question,
            llm=llm,
            use_tools=True,
            run_python=run_python,
            use_pot_ranking=True,
        )

    assert prediction == Prediction("q1", "B")
    ranking.assert_called_once_with(
        llm,
        question.question,
        question.choices,
        run_python=run_python,
    )
    default_pot.assert_not_called()

    with (
        patch("src.pipeline.solve_with_tools_ranking", return_value="B") as ranking,
        patch("src.pipeline.solve_with_tools", return_value="A") as default_pot,
    ):
        prediction = answer_question(
            question,
            llm=llm,
            use_tools=True,
            run_python=run_python,
            use_pot_ranking=False,
        )

    assert prediction == Prediction("q1", "A")
    ranking.assert_not_called()
    default_pot.assert_called_once_with(
        llm,
        question.question,
        question.choices,
        run_python,
    )


def test_pot_ranking_cli_uses_none_sentinel() -> None:
    assert parse_args([]).pot_ranking is None
    assert parse_args(["--pot-ranking"]).pot_ranking is True
    assert parse_args(["--no-pot-ranking"]).pot_ranking is False
