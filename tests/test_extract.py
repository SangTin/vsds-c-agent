import pytest

from src.extract import extract_letter, letter_set, validate_letter


@pytest.mark.parametrize(
    ("n_choices", "expected"),
    [(0, ""), (2, "AB"), (11, "ABCDEFGHIJK"), (30, "ABCDEFGHIJKLMNOPQRSTUVWXYZ")],
)
def test_letter_set(n_choices: int, expected: str) -> None:
    assert letter_set(n_choices) == expected


@pytest.mark.parametrize(
    ("letter", "n_choices", "fallback", "expected"),
    [
        ("B", 4, "A", "B"),
        ("Z", 4, "A", "A"),
        ("b", 4, "A", "B"),
        (" C ", 4, "A", "C"),
        ("Z", 4, "b", "B"),
        ("Z", 4, "Z", "A"),
    ],
)
def test_validate_letter(
    letter: str, n_choices: int, fallback: str, expected: str
) -> None:
    assert validate_letter(letter, n_choices, fallback) == expected


@pytest.mark.parametrize(
    ("text", "n_choices", "fallback", "expected"),
    [
        ("A", 4, "A", "A"),
        ("A.", 4, "A", "A"),
        ("(A)", 4, "A", "A"),
        ("Đáp án: B", 4, "A", "B"),
        ("No answer here", 4, "C", "C"),
        ("E", 4, "B", "B"),
    ],
)
def test_extract_letter(
    text: str, n_choices: int, fallback: str, expected: str
) -> None:
    assert extract_letter(text, n_choices, fallback) == expected
