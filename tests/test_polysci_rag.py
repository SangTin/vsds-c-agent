from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock, patch

from entrypoint import load_config, parse_args
from src.pipeline import answer_question
from src.router import is_polysci_question
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


def test_is_polysci_question_detects_five_subject_cues() -> None:
    assert is_polysci_question(
        "Theo tư tưởng Hồ Chí Minh, vai trò của Đảng Cộng sản là gì?",
        ["A", "B"],
    )
    assert is_polysci_question(
        "Chủ nghĩa Mác-Lênin về giai cấp công nhân nêu nội dung nào?",
        ["A", "B"],
    )
    assert is_polysci_question(
        "Theo Triết Mác-Lênin, quy luật phủ định của phủ định thuộc nội dung nào?",
        ["A", "B"],
    )
    assert is_polysci_question(
        "Học thuyết giá trị thặng dư của Mác giải thích vấn đề gì?",
        ["A", "B"],
    )


def test_is_polysci_question_rejects_non_polysci_and_passage_questions() -> None:
    assert not is_polysci_question("Tính 6*7 bằng bao nhiêu?", ["41", "42"])
    assert not is_polysci_question(
        "Theo Bộ luật Dân sự, điều 117 quy định điều kiện gì?",
        ["A", "B"],
    )
    assert not is_polysci_question(
        "Đoạn thông tin:\nHồ Chí Minh được nhắc trong bài.\nCâu hỏi: Nội dung chính là gì?",
        ["Tư tưởng Hồ Chí Minh", "Văn hóa"],
    )


def test_pipeline_uses_polysci_retriever_and_injects_context() -> None:
    question = Question(
        "q1",
        "Theo tư tưởng Hồ Chí Minh, vai trò của Đảng Cộng sản là gì?",
        ["A", "B"],
    )
    llm = MockLLM(direct_answer="A", mcq_answer="B")
    retriever = Mock()
    retriever.retrieve.return_value = [
        {
            "doc_id": "hcm",
            "title": "Tư tưởng Hồ Chí Minh",
            "chunk_id": 2,
            "text": "x" * 1600,
            "score": 0.84,
        }
    ]

    prediction = answer_question(
        question,
        llm=llm,
        polysci_retriever=retriever,
        polysci_min_score=0.7,
    )

    assert prediction == Prediction("q1", "B")
    retriever.retrieve.assert_called_once_with(question.question)
    assert llm.answer_calls == []
    context = llm.answer_mcq_calls[0]["context"]
    assert "Trích giáo trình lý luận chính trị:" in context
    assert "[1] Tư tưởng Hồ Chí Minh:" in context
    assert "x" * 1500 in context
    assert "x" * 1501 not in context


def test_pipeline_below_polysci_threshold_falls_through_to_direct() -> None:
    question = Question(
        "q1",
        "Chủ nghĩa Mác-Lênin về giai cấp công nhân nêu nội dung nào?",
        ["A", "B"],
    )
    llm = MockLLM(direct_answer="A", mcq_answer="B")
    retriever = Mock()
    retriever.retrieve.return_value = [
        {"title": "Chủ nghĩa xã hội khoa học", "text": "context", "score": 0.2}
    ]

    prediction = answer_question(
        question,
        llm=llm,
        polysci_retriever=retriever,
        polysci_min_score=0.7,
    )

    assert prediction == Prediction("q1", "A")
    assert llm.answer_mcq_calls == []
    assert llm.answer_calls == [
        {"question": question.question, "choices": question.choices, "retrieved": None}
    ]


def test_pipeline_polysci_exception_falls_back_to_direct() -> None:
    question = Question(
        "q1",
        "Học thuyết giá trị thặng dư của Mác giải thích vấn đề gì?",
        ["A", "B", "C"],
    )
    llm = MockLLM(direct_answer="C", mcq_answer="B")
    retriever = Mock()
    retriever.retrieve.side_effect = RuntimeError("search failed")

    prediction = answer_question(question, llm=llm, polysci_retriever=retriever)

    assert prediction == Prediction("q1", "C")
    assert llm.answer_mcq_calls == []
    assert llm.answer_calls == [
        {"question": question.question, "choices": question.choices, "retrieved": None}
    ]


def test_pipeline_legal_priority_over_polysci_when_both_match() -> None:
    question = Question(
        "q1",
        "Theo Luật Giáo dục, tư tưởng Hồ Chí Minh có vai trò gì?",
        ["A", "B"],
    )
    llm = MockLLM(direct_answer="A", mcq_answer="B")
    legal_retriever = Mock()
    legal_retriever.retrieve.return_value = [
        {"title": "Luật Giáo dục", "text": "legal context", "score": 0.9}
    ]
    polysci_retriever = Mock()
    polysci_retriever.retrieve.return_value = [
        {"title": "Tư tưởng Hồ Chí Minh", "text": "polysci context", "score": 0.9}
    ]

    prediction = answer_question(
        question,
        llm=llm,
        legal_retriever=legal_retriever,
        legal_min_score=0.7,
        polysci_retriever=polysci_retriever,
        polysci_min_score=0.7,
    )

    assert prediction == Prediction("q1", "B")
    legal_retriever.retrieve.assert_called_once_with(question.question)
    polysci_retriever.retrieve.assert_not_called()
    assert "Trích văn bản pháp luật liên quan:" in llm.answer_mcq_calls[0]["context"]


def test_polysci_cli_uses_none_sentinel_and_supports_overrides() -> None:
    assert parse_args([]).polysci_rag is None
    assert parse_args(["--polysci-rag"]).polysci_rag is True
    assert parse_args(["--no-polysci-rag"]).polysci_rag is False
    assert parse_args(["--polysci-device", "cuda"]).polysci_device == "cuda"


def test_load_config_reads_polysci_rag_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    yaml = ModuleType("yaml")
    yaml.safe_load = Mock(  # type: ignore[attr-defined]
        return_value={
            "polysci_rag": {
                "enabled": True,
                "index_path": "custom/polysci/index.faiss",
                "metadata_path": "custom/polysci/metadata.jsonl",
                "top_k": 5,
                "min_score": 0.75,
                "device": "cuda",
                "model_name": "custom/model",
            }
        }
    )

    with patch.dict(sys.modules, {"yaml": yaml}):
        polysci_config = load_config(config_path)["polysci_rag"]

    assert polysci_config == {
        "enabled": True,
        "index_path": "custom/polysci/index.faiss",
        "metadata_path": "custom/polysci/metadata.jsonl",
        "top_k": 5,
        "min_score": 0.75,
        "device": "cuda",
        "model_name": "custom/model",
    }
