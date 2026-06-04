import sys

from src.extract import validate_letter
from src.llm import LLMAnswerer
from src.schema import Prediction, Question


def answer_question(
    q: Question,
    fallback: str = "A",
    llm: LLMAnswerer | None = None,
) -> Prediction:
    """Answer one question with the LLM, falling back without breaking the batch."""
    fallback_letter = validate_letter(fallback, len(q.choices))
    if llm is None:
        return Prediction(q.qid, fallback_letter)

    try:
        answer = llm.answer_mcq(q.question, q.choices)
        validated = validate_letter(answer, len(q.choices), fallback_letter)
        return Prediction(q.qid, validated)
    except Exception as exc:
        print(
            f"Warning: LLM failed for {q.qid!r}; falling back to "
            f"{fallback_letter}: {exc}",
            file=sys.stderr,
        )
        return Prediction(q.qid, fallback_letter)
