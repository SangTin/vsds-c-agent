from __future__ import annotations

import re
import unicodedata


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
_LAW_ARTICLE_RE = re.compile(r"\b(?:điều|dieu)\s+\d+", flags=re.IGNORECASE)
_LAW_KEYWORD_RE = re.compile(
    r"\b(?:"
    r"bộ luật|bo luat|luật|luat|khoản|khoan|nghị định|nghi dinh|"
    r"thông tư|thong tu|hiến pháp|hien phap|tội|hình sự|hinh su|"
    r"dân sự|dan su|nghĩa vụ|nghia vu|trách nhiệm hình sự|"
    r"trach nhiem hinh su|xử phạt|xu phat|vi phạm|vi pham|"
    r"pháp luật|phap luat|hợp đồng|hop dong"
    r")\b",
    flags=re.IGNORECASE,
)
_PASSAGE_MARKERS = ("đoạn thông tin:", "doan thong tin:")
_HOW_TO_RE = re.compile(
    r"làm thế nào|cách nào|phương thức|bằng cách nào|làm sao|"
    r"cách hiệu quả|như thế nào|"
    r"hành động nào|hành vi nào|"
    r"thực hiện .{0,20}(?:nào|ra sao|thế nào)",
    flags=re.IGNORECASE,
)
_VIOLATION_RE = re.compile(
    r"vi phạm|trái pháp luật|trái quy định|phá hoại|tham nhũng|làm giả|"
    r"lợi dụng chức vụ|lợi dụng quyền|xuyên tạc|bịa đặt|"
    r"phát tán tài liệu mật|phát tán thông tin sai|"
    r"xúc phạm(?:\s+các)?\s+(?:biểu tượng|lãnh tụ)|"
    r"tạo ra văn bản trái|làm suy yếu|công chức.*tham nhũng|"
    r"phá hoại tinh thần",
    flags=re.IGNORECASE,
)
_REFUSAL_RE = re.compile(
    r"tôi không thể|không thể trả lời|không thể cung cấp",
    flags=re.IGNORECASE,
)
_POLYSCI_KEYWORD_RE = re.compile(
    r"\b(?:"
    r"ho chi minh|tu tuong|mac|lenin|angghen|"
    r"chu nghia|duy vat|bien chung|"
    r"kinh te chinh tri|gia tri thang du|tich luy tu ban|"
    r"chu nghia xa hoi|chu nghia cong san|"
    r"cach mang|giai cap|vo san|"
    r"dang cong san|dang ta|cuong linh|dai hoi dang|"
    r"phuong thuc san xuat|luc luong san xuat|quan he san xuat|"
    r"hinh thai kinh te|duy vat lich su|duy vat bien chung|"
    r"y thuc xa hoi|ton tai xa hoi"
    r")\b",
    flags=re.IGNORECASE,
)


def _fold_vietnamese(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(
        char for char in decomposed if unicodedata.category(char) != "Mn"
    ).casefold()


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


def is_law_question(question: str, choices: list[str]) -> bool:
    """Return True for questions likely answered by Vietnamese legal texts.

    Precision over recall: legal RAG is only useful when the source corpus is
    likely to contain the answer. Passage questions already include their own
    evidence, and broad non-legal uses such as "quy luật" should stay direct.
    """
    question_text = question.casefold()
    if any(marker in question_text for marker in _PASSAGE_MARKERS):
        return False

    text = "\n".join([question, *choices]).casefold()
    if _LAW_ARTICLE_RE.search(text):
        return True
    match = _LAW_KEYWORD_RE.search(text)
    if not match:
        return False
    if match.group(0) in {"luật", "luat"} and text[
        max(0, match.start() - 4) : match.start()
    ] == "quy ":
        return False
    return True


def detect_alignment_bait(question: str, choices: list[str]) -> str | None:
    """Return the refusal letter if question is an alignment bait, else None."""
    ql = question.lower()
    if not _HOW_TO_RE.search(ql):
        return None
    if not _VIOLATION_RE.search(ql):
        return None
    for i, choice in enumerate(choices):
        if _REFUSAL_RE.search(choice.lower()):
            return chr(65 + i)
    return None


def is_polysci_question(question: str, choices: list[str]) -> bool:
    """Return True for the five mandatory political-theory subjects.

    Precision over recall: the polysci index is a narrow textbook corpus, so we
    route only named authors/subjects or canonical Marxist-Leninist concepts.
    Passage questions already contain their own evidence and stay direct.
    """
    question_text = question.casefold()
    if any(marker in question_text for marker in _PASSAGE_MARKERS):
        return False

    text = _fold_vietnamese("\n".join([question, *choices]))
    return _POLYSCI_KEYWORD_RE.search(text) is not None
