from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from src.agent.loop import solve_with_tools
from src.extract import validate_letter
from src.llm import LLMAnswerer
from src.rag.retriever import FaissRetriever
from src.router import is_math_question
from src.schema import Prediction, Question


def answer_question(
    q: Question,
    fallback: str = "A",
    llm: LLMAnswerer | None = None,
    retriever: FaissRetriever | None = None,
    use_tools: bool = False,
    run_python: Callable[[str], dict[str, Any]] | None = None,
) -> Prediction:
    """Answer one question with the LLM, falling back without breaking the batch."""
    fallback_letter = validate_letter(fallback, len(q.choices))
    if (
        use_tools
        and llm is not None
        and run_python is not None
        and is_math_question(q.question, q.choices)
    ):
        try:
            letter = solve_with_tools(llm, q.question, q.choices, run_python)
            if letter:
                return Prediction(
                    q.qid,
                    validate_letter(letter, len(q.choices), fallback_letter),
                )
        except Exception as exc:
            print(
                f"Warning: tool path failed for {q.qid!r}; "
                f"falling back to direct answer: {exc}",
                file=sys.stderr,
            )

    retrieved: list[dict[str, Any]] | None = None
    should_retrieve = retriever is not None and "Đoạn thông tin:" not in q.question
    if should_retrieve:
        try:
            retrieved = retriever.retrieve(q.question)
        except Exception as exc:
            print(
                f"Warning: RAG retrieval failed for {q.qid!r}; "
                f"continuing without retrieved context: {exc}",
                file=sys.stderr,
            )

    if llm is None:
        return Prediction(q.qid, fallback_letter)

    try:
        answer = llm.answer(q.question, q.choices, retrieved=retrieved)
        validated = validate_letter(answer, len(q.choices), fallback_letter)
        return Prediction(q.qid, validated)
    except Exception as exc:
        print(
            f"Warning: LLM failed for {q.qid!r}; falling back to "
            f"{fallback_letter}: {exc}",
            file=sys.stderr,
        )
        return Prediction(q.qid, fallback_letter)
