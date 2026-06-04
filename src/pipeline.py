import sys
from typing import Any

from src.extract import validate_letter
from src.llm import LLMAnswerer
from src.rag.retriever import FaissRetriever
from src.schema import Prediction, Question


def answer_question(
    q: Question,
    fallback: str = "A",
    llm: LLMAnswerer | None = None,
    retriever: FaissRetriever | None = None,
) -> Prediction:
    """Answer one question with the LLM, falling back without breaking the batch."""
    fallback_letter = validate_letter(fallback, len(q.choices))
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
