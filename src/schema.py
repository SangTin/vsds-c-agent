from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    choices: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.qid, str) or not self.qid.strip():
            raise ValueError("qid must be a non-empty string")
        if (
            not isinstance(self.choices, list)
            or not self.choices
            or not all(isinstance(choice, str) for choice in self.choices)
        ):
            raise ValueError("choices must be a non-empty list of strings")


@dataclass(frozen=True)
class Prediction:
    qid: str
    answer: str

    def __post_init__(self) -> None:
        if not isinstance(self.qid, str) or not self.qid.strip():
            raise ValueError("qid must be a non-empty string")
        if (
            not isinstance(self.answer, str)
            or len(self.answer) != 1
            or self.answer not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        ):
            raise ValueError("answer must be one uppercase letter from A to Z")
