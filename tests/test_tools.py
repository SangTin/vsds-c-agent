from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock, patch

from entrypoint import load_config, parse_args
from src.agent.loop import solve_with_tools
from src.pipeline import answer_question
from src.router import is_math_question
from src.schema import Prediction, Question
from src.tools.sandbox import run_python


class MockLLM:
    seed = 123

    def __init__(self, completions: list[str], letter: str = "B") -> None:
        self.completions = completions
        self.letter = letter
        self.complete_calls: list[dict[str, Any]] = []
        self.answer_mcq_calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self.complete_calls.append({"messages": messages, "kwargs": kwargs})
        return self.completions.pop(0)

    def answer_mcq(
        self,
        question: str,
        choices: list[str],
        context: str | None = None,
    ) -> str:
        self.answer_mcq_calls.append(
            {"question": question, "choices": choices, "context": context}
        )
        return self.letter


def test_sandbox_run_python_prints_stdout() -> None:
    result = run_python("print(6*7)")

    assert result["ok"] is True
    assert result["stdout"] == "42"
    assert result["stderr"] == ""


def test_sandbox_run_python_returns_stderr_on_exception() -> None:
    result = run_python("raise ValueError('boom')")

    assert result["ok"] is False
    assert "ValueError" in str(result["stderr"])
    assert "boom" in str(result["stderr"])


def test_sandbox_run_python_kills_infinite_loop() -> None:
    result = run_python("while True:\n    pass", timeout=0.2)

    assert result["ok"] is False
    assert "Timeout" in str(result["stderr"])


def test_router_detects_math_notation() -> None:
    assert is_math_question(
        "Tính giá trị của $\\int_0^1 x^2 dx$",
        ["0", "1/3", "1/2", "1"],
    )


def test_router_detects_gdp_quantity_question() -> None:
    assert is_math_question(
        "GDP của Việt Nam năm 2023 là bao nhiêu tỷ USD?",
        ["300", "430", "900", "1200"],
    )


def test_router_detects_numeric_choices() -> None:
    assert is_math_question(
        "Chọn đáp án đúng.",
        ["1/3", "2/3", "3/4", "5/6"],
    )


def test_router_avoids_non_math_factual_question() -> None:
    assert not is_math_question(
        "Tư tưởng Hồ Chí Minh có nguồn gốc từ đâu?",
        ["Chủ nghĩa Mác-Lênin", "Văn học dân gian", "Địa lý tự nhiên", "Kinh tế học"],
    )


def test_router_avoids_reading_comprehension_question() -> None:
    assert not is_math_question(
        "Đoạn thông tin:\nLan đi học mỗi ngày.\nCâu hỏi: Nhân vật chính là ai?",
        ["Lan", "Hoa", "Nam", "An"],
    )


def test_tools_cli_uses_none_sentinel_and_supports_explicit_false() -> None:
    assert parse_args([]).tools is None
    assert parse_args(["--tools"]).tools is True
    assert parse_args(["--no-tools"]).tools is False
    assert parse_args(["--tool-timeout", "1.5"]).tool_timeout == 1.5


def test_load_config_reads_tool_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    yaml = ModuleType("yaml")
    yaml.safe_load = Mock(  # type: ignore[attr-defined]
        return_value={"model": {"use_tools": True, "tool_timeout": 2.5}}
    )

    with patch.dict(sys.modules, {"yaml": yaml}):
        model_config = load_config(config_path)["model"]

    assert model_config["use_tools"] is True
    assert model_config["tool_timeout"] == 2.5


def test_agent_extracts_python_block_runs_tool_and_selects_letter() -> None:
    llm = MockLLM(["```python\nprint(6*7)\n```"], letter="B")
    run_calls: list[str] = []

    def fake_run_python(code: str) -> dict[str, Any]:
        run_calls.append(code)
        return {"ok": True, "stdout": "42", "stderr": ""}

    answer = solve_with_tools(
        llm,
        "Tính 6*7.",
        ["41", "42"],
        fake_run_python,
    )

    assert answer == "B"
    assert run_calls == ["print(6*7)"]
    assert llm.complete_calls[0]["kwargs"] == {
        "max_tokens": 256,
        "temperature": 0.0,
        "seed": 123,
    }
    assert "Kết quả tính toán: 42" in llm.answer_mcq_calls[0]["context"]


def test_agent_retries_once_after_tool_failure() -> None:
    llm = MockLLM(
        [
            "```python\nprint(1/0)\n```",
            "```\nprint(42)\n```",
        ],
        letter="A",
    )
    run_calls: list[str] = []

    def fake_run_python(code: str) -> dict[str, Any]:
        run_calls.append(code)
        if len(run_calls) == 1:
            return {"ok": False, "stdout": "", "stderr": "ZeroDivisionError"}
        return {"ok": True, "stdout": "42", "stderr": ""}

    answer = solve_with_tools(
        llm,
        "Tính 6*7.",
        ["42", "41"],
        fake_run_python,
    )

    assert answer == "A"
    assert run_calls == ["print(1/0)", "print(42)"]
    assert len(llm.complete_calls) == 2
    retry_prompt = llm.complete_calls[1]["messages"][-1]["content"]
    assert "ZeroDivisionError" in retry_prompt
    assert "print(1/0)" in retry_prompt


def test_pipeline_routes_math_question_into_tools() -> None:
    question = Question("q1", "Tính 6*7 bằng bao nhiêu?", ["41", "42"])
    llm = Mock()
    run_python_mock = Mock()

    with patch("src.pipeline.solve_with_tools", return_value="B") as solve:
        prediction = answer_question(
            question,
            llm=llm,
            use_tools=True,
            run_python=run_python_mock,
        )

    assert prediction == Prediction("q1", "B")
    solve.assert_called_once_with(llm, question.question, question.choices, run_python_mock)
    llm.answer.assert_not_called()


def test_pipeline_routes_non_math_question_into_plain_llm() -> None:
    question = Question("q1", "Tư tưởng Hồ Chí Minh có nguồn gốc từ đâu?", ["A", "B"])
    llm = Mock()
    llm.answer.return_value = "A"
    run_python_mock = Mock()

    with patch("src.pipeline.solve_with_tools") as solve:
        prediction = answer_question(
            question,
            llm=llm,
            use_tools=True,
            run_python=run_python_mock,
        )

    assert prediction == Prediction("q1", "A")
    solve.assert_not_called()
    llm.answer.assert_called_once_with(
        question.question,
        question.choices,
        retrieved=None,
    )


def test_pipeline_tool_exception_falls_back_without_crashing_batch() -> None:
    question = Question("q1", "Tính 6*7 bằng bao nhiêu?", ["41", "42"])
    llm = Mock()
    llm.answer.side_effect = RuntimeError("plain llm failed")

    with patch("src.pipeline.solve_with_tools", side_effect=RuntimeError("tool failed")):
        prediction = answer_question(
            question,
            fallback="B",
            llm=llm,
            use_tools=True,
            run_python=Mock(),
        )

    assert prediction == Prediction("q1", "B")
