from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

from src.pipeline import answer_question
from src.router import is_law_question
from src.schema import Prediction, Question


class MockLLM:
    def __init__(self, direct_answer: str = "A", mcq_answer: str = "B") -> None:
        self.direct_answer = direct_answer
        self.mcq_answer = mcq_answer
        self.answer_calls: list[dict[str, Any]] = []
        self.answer_mcq_calls: list[dict[str, Any]] = []

    def answer(
        self,
        question: str,
        choices: list[str],
        *,
        retrieved: list[dict[str, Any]] | None = None,
    ) -> str:
        self.answer_calls.append(
            {"question": question, "choices": choices, "retrieved": retrieved}
        )
        return self.direct_answer

    def answer_mcq(
        self,
        question: str,
        choices: list[str],
        context: str | None = None,
    ) -> str:
        self.answer_mcq_calls.append(
            {"question": question, "choices": choices, "context": context}
        )
        return self.mcq_answer


def test_is_law_question_detects_legal_cues() -> None:
    assert is_law_question(
        "Theo Bộ luật Hình sự, độ tuổi chịu trách nhiệm hình sự là bao nhiêu?",
        ["14 tuổi", "16 tuổi"],
    )
    assert is_law_question(
        "Luật BVMT 2020 có bao nhiêu nguyên tắc?",
        ["4", "6"],
    )


def test_is_law_question_rejects_non_law_and_passage_questions() -> None:
    assert not is_law_question("Tính 6*7 bằng bao nhiêu?", ["41", "42"])
    assert not is_law_question(
        "Tư tưởng Hồ Chí Minh có nguồn gốc từ đâu?",
        ["Chủ nghĩa Mác-Lênin", "Văn học dân gian"],
    )
    assert not is_law_question(
        "Đoạn thông tin:\nLuật được nhắc trong bài.\nCâu hỏi: Nội dung chính là gì?",
        ["A", "B"],
    )


def test_pipeline_uses_legal_retriever_and_injects_legal_context() -> None:
    question = Question(
        "q1",
        "Theo Bộ luật Hình sự, độ tuổi chịu trách nhiệm hình sự là bao nhiêu?",
        ["14 tuổi", "16 tuổi"],
    )
    llm = MockLLM(direct_answer="A", mcq_answer="B")
    retriever = Mock()
    retriever.retrieve.return_value = [
        {
            "doc_id": "blhs",
            "title": "Bộ luật Hình sự 2015",
            "chunk_id": 3,
            "text": "x" * 1600,
            "score": 0.82,
        }
    ]

    prediction = answer_question(
        question,
        llm=llm,
        legal_retriever=retriever,
        legal_min_score=0.7,
    )

    assert prediction == Prediction("q1", "B")
    retriever.retrieve.assert_called_once_with(question.question)
    assert llm.answer_calls == []
    context = llm.answer_mcq_calls[0]["context"]
    assert "Trích văn bản pháp luật liên quan:" in context
    assert "[1] Bộ luật Hình sự 2015:" in context
    assert "x" * 1500 in context
    assert "x" * 1501 not in context


def test_pipeline_below_legal_threshold_falls_through_to_direct() -> None:
    question = Question(
        "q1",
        "Theo Bộ luật Hình sự, độ tuổi chịu trách nhiệm hình sự là bao nhiêu?",
        ["14 tuổi", "16 tuổi"],
    )
    llm = MockLLM(direct_answer="A", mcq_answer="B")
    retriever = Mock()
    retriever.retrieve.return_value = [
        {"title": "Bộ luật Hình sự", "text": "context", "score": 0.2}
    ]

    prediction = answer_question(
        question,
        llm=llm,
        legal_retriever=retriever,
        legal_min_score=0.7,
    )

    assert prediction == Prediction("q1", "A")
    assert llm.answer_mcq_calls == []
    assert llm.answer_calls == [
        {"question": question.question, "choices": question.choices, "retrieved": None}
    ]


def test_pipeline_math_question_still_routes_to_tools_before_legal_rag() -> None:
    question = Question("q1", "Tính 6*7 bằng bao nhiêu?", ["41", "42"])
    llm = MockLLM()
    run_python = Mock()
    legal_retriever = Mock()

    with patch("src.pipeline.solve_with_tools", return_value="B") as solve:
        prediction = answer_question(
            question,
            llm=llm,
            use_tools=True,
            run_python=run_python,
            legal_retriever=legal_retriever,
        )

    assert prediction == Prediction("q1", "B")
    solve.assert_called_once_with(llm, question.question, question.choices, run_python)
    legal_retriever.retrieve.assert_not_called()
    assert llm.answer_calls == []
    assert llm.answer_mcq_calls == []


def test_pipeline_legal_exception_falls_back_to_direct() -> None:
    question = Question(
        "q1",
        "Theo Bộ luật Dân sự, hợp đồng có hiệu lực khi nào?",
        ["A", "B", "C"],
    )
    llm = MockLLM(direct_answer="C", mcq_answer="B")
    retriever = Mock()
    retriever.retrieve.side_effect = RuntimeError("search failed")

    prediction = answer_question(question, llm=llm, legal_retriever=retriever)

    assert prediction == Prediction("q1", "C")
    assert llm.answer_mcq_calls == []
    assert llm.answer_calls == [
        {"question": question.question, "choices": question.choices, "retrieved": None}
    ]
