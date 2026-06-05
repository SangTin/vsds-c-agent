from __future__ import annotations

import re


_LATEX_RE = re.compile(
    r"\$[^$]+\$|\\(?:frac|int|sum|sqrt)|(?<!\w)sqrt\s*\(|[A-Za-z0-9)}]\s*\^\s*[-+{(A-Za-z0-9]",
    flags=re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?(?:\s*/\s*\d+(?:[.,]\d+)?)?")
_QUANTITY_CHOICE_RE = re.compile(
    r"^\s*(?:khoảng|xấp xỉ|gần|hơn|dưới|trên|từ)?\s*"
    r"[-+]?(?:\d+(?:[.,]\d+)?(?:\s*/\s*\d+(?:[.,]\d+)?)?|\d+(?:[.,]\d+)?\s*[eE][-+]?\d+)"
    r"\s*(?:%|cm|mm|m|km|kg|g|tấn|m/s|km/h|usd|đô la|đồng|vnd|tỷ|triệu|nghìn|ha|m2|m\^2)?\s*$",
    flags=re.IGNORECASE,
)
_DIRECT_MATH_KEYWORDS = (
    # "giá trị" alone is a false friend ("giá trị tư tưởng/văn hóa"); real math
    # phrasing ("tính giá trị", LaTeX, numeric choices) is caught by other rules.
    "tính",
    "phương trình",
    "đạo hàm",
    "tích phân",
    "xác suất",
    "lãi suất",
    "phần trăm",
    "tỷ lệ",
)


def _mostly_numeric_choices(choices: list[str]) -> bool:
    if not choices:
        return False
    numeric_count = sum(
        1
        for choice in choices
        if _QUANTITY_CHOICE_RE.search(choice) or (
            len(choice.strip()) <= 32 and _NUMBER_RE.search(choice)
        )
    )
    return numeric_count > len(choices) / 2


def is_math_question(question: str, choices: list[str]) -> bool:
    """Return True when a question is likely better answered by computation.

    Precision over recall: only strong, unambiguous calculation signals route
    to Program-of-Thought. Reading-comprehension questions (which carry a
    passage and incidental numbers) and bare factual-count questions are kept
    on the direct path — generating code for them produces a spurious number
    that would corrupt the answer, the same failure mode seen with RAG.
    """
    text = question.casefold()
    # Reading-comprehension items embed their own passage; their incidental
    # numbers/units must not trigger the tool path.
    if "đoạn thông tin:" in text or len(question) > 500:
        return False
    if _mostly_numeric_choices(choices):
        return True
    if _LATEX_RE.search(question):
        return True
    if any(keyword in text for keyword in _DIRECT_MATH_KEYWORDS):
        return True
    return False
