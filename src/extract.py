import re


def letter_set(n_choices: int) -> str:
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_choices]


def validate_letter(letter: str, n_choices: int, fallback: str = "A") -> str:
    valid_letters = letter_set(n_choices)
    normalized = letter.strip().upper()
    if len(normalized) == 1 and normalized in valid_letters:
        return normalized

    normalized_fallback = fallback.strip().upper()
    if len(normalized_fallback) == 1 and normalized_fallback in valid_letters:
        return normalized_fallback
    return "A"


def extract_letter(text: str, n_choices: int, fallback: str = "A") -> str:
    valid_letters = letter_set(n_choices)
    if valid_letters:
        match = re.search(
            rf"(?<!\w)([{re.escape(valid_letters)}])(?!\w)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()
    return validate_letter("", n_choices, fallback)
