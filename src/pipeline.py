from src.extract import validate_letter
from src.schema import Prediction, Question


def answer_question(q: Question, fallback: str = "A") -> Prediction:
    """Return the M1 stub answer; Milestone 2 will plug in the LLM."""
    return Prediction(q.qid, validate_letter(fallback, len(q.choices)))
