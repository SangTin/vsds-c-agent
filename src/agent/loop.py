from __future__ import annotations

import re
import sys
from collections.abc import Callable
from typing import Any

from src.extract import letter_set, validate_letter


RunPython = Callable[[str], dict[str, Any]]

_PYTHON_BLOCK_RE = re.compile(r"```(?:python|py)\s*\n?(.*?)```", flags=re.IGNORECASE | re.DOTALL)
_ANY_BLOCK_RE = re.compile(r"```\s*\n?(.*?)```", flags=re.DOTALL)
_RANKING_INSTRUCTION = (
    "Đây là câu hỏi trắc nghiệm cần tính toán. Hãy viết một đoạn Python ngắn "
    "thay từng giá trị trong các đáp án A, B, C... vào công thức/biểu thức trong câu hỏi, "
    "tính kết quả từng đáp án, so sánh với mục tiêu hoặc điều kiện đề bài, "
    "rồi in ra chữ cái duy nhất ứng với đáp án đúng nhất.\n\n"
    "Quy ước: dùng print(letter) ở cuối, với letter là một chữ cái viết hoa A-J. "
    "Không in giải thích."
)


def _choice_lines(choices: list[str]) -> str:
    return "\n".join(
        f"{letter}. {choice}"
        for letter, choice in zip(letter_set(len(choices)), choices)
    )


def _extract_code(text: str) -> str:
    match = _PYTHON_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    match = _ANY_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _initial_messages(question: str, choices: list[str]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Bạn là trợ lý giải bài trắc nghiệm bằng tính toán. "
                "Chỉ viết mã Python để tính đáp số, không giải thích."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Câu hỏi:\n{question}\n\n"
                f"Lựa chọn:\n{_choice_lines(choices)}\n\n"
                "Viết một đoạn mã Python ngắn, dùng sympy nếu cần, tính toán "
                "và in ra kết quả cuối cùng bằng print(). Chỉ in giá trị/đáp số, "
                "không giải thích. Trả lời bằng một khối mã ```python ... ```."
            ),
        },
    ]


def _retry_messages(
    question: str,
    choices: list[str],
    previous_response: str,
    previous_code: str,
    error: str,
) -> list[dict[str, str]]:
    messages = _initial_messages(question, choices)
    messages.append({"role": "assistant", "content": previous_response})
    messages.append(
        {
            "role": "user",
            "content": (
                "Đoạn mã trước không chạy được hoặc không in ra kết quả.\n\n"
                f"Mã trước:\n```python\n{previous_code}\n```\n\n"
                f"Lỗi/kết quả:\n{error}\n\n"
                "Hãy sửa mã. Vẫn chỉ trả lời bằng một khối mã Python và phải print() đáp số cuối cùng."
            ),
        }
    )
    return messages


def _ranking_messages(question: str, choices: list[str]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Bạn là trợ lý giải bài trắc nghiệm bằng tính toán. "
                "Chỉ viết mã Python để chọn đáp án, không giải thích."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Câu hỏi:\n{question}\n\n"
                f"Lựa chọn:\n{_choice_lines(choices)}\n\n"
                f"{_RANKING_INSTRUCTION}\n\n"
                "Trả lời bằng một khối mã ```python ... ```."
            ),
        },
    ]


def _ranking_retry_messages(
    question: str,
    choices: list[str],
    previous_response: str,
    previous_code: str,
    error: str,
) -> list[dict[str, str]]:
    messages = _ranking_messages(question, choices)
    messages.append({"role": "assistant", "content": previous_response})
    messages.append(
        {
            "role": "user",
            "content": (
                "Đoạn mã trước không chạy được hoặc không in ra đúng một chữ cái đáp án.\n\n"
                f"Mã trước:\n```python\n{previous_code}\n```\n\n"
                f"Lỗi/kết quả:\n{error}\n\n"
                "Hãy sửa mã. Vẫn chỉ trả lời bằng một khối mã Python và phải print() "
                "duy nhất một chữ cái viết hoa A-J."
            ),
        }
    )
    return messages


def _complete_code(llm: Any, messages: list[dict[str, str]]) -> str:
    return llm.complete(
        messages,
        max_tokens=256,
        temperature=0.0,
        seed=getattr(llm, "seed", None),
    )


def _valid_ranking_letter(output: str, n_choices: int) -> str | None:
    normalized = output.strip().upper()
    if len(normalized) != 1:
        return None
    if normalized not in letter_set(n_choices):
        return None
    validated = validate_letter(normalized, n_choices, fallback="")
    if validated != normalized:
        return None
    return normalized


def solve_with_tools_ranking(
    llm: Any,
    question: str,
    choices: list[str],
    *,
    run_python: RunPython,
    max_retries: int = 1,
) -> str | None:
    messages = _ranking_messages(question, choices)
    last_response = ""
    last_code = ""
    attempts = max(1, max_retries + 1)

    for attempt in range(attempts):
        try:
            response = _complete_code(llm, messages)
            code = _extract_code(response)
        except Exception as exc:
            response = ""
            code = ""
            result = {"ok": False, "stdout": "", "stderr": str(exc)}
        else:
            if not code:
                result = {
                    "ok": False,
                    "stdout": "",
                    "stderr": "Model did not produce code",
                }
            else:
                try:
                    result = run_python(code)
                except Exception as exc:
                    result = {"ok": False, "stdout": "", "stderr": str(exc)}

        last_response = response
        last_code = code
        stdout = str(result.get("stdout", "")).strip()
        stderr = str(result.get("stderr", "")).strip()
        if result.get("ok", True):
            letter = _valid_ranking_letter(stdout, len(choices))
            if letter is not None:
                return letter

        if attempt + 1 >= attempts:
            break
        error = stderr or stdout or "Không có stdout."
        messages = _ranking_retry_messages(
            question,
            choices,
            previous_response=last_response,
            previous_code=last_code,
            error=error,
        )

    return None


def solve_with_tools(
    llm: Any,
    question: str,
    choices: list[str],
    run_python: RunPython,
    max_attempts: int = 2,
    verbose: bool = False,
) -> str | None:
    messages = _initial_messages(question, choices)
    last_response = ""
    last_code = ""
    attempts = max(1, max_attempts)

    for attempt in range(attempts):
        response = _complete_code(llm, messages)
        code = _extract_code(response)
        last_response = response
        last_code = code
        if not code:
            result = {"ok": False, "stdout": "", "stderr": "Model did not produce code"}
        else:
            result = run_python(code)

        stdout = str(result.get("stdout", "")).strip()
        stderr = str(result.get("stderr", "")).strip()
        if result.get("ok") and stdout:
            context = (
                f"Kết quả tính toán: {stdout}\n"
                "Dựa trên kết quả trên, chọn đáp án đúng. Trả lời bằng đúng 1 chữ cái."
            )
            try:
                return llm.answer_mcq(question, choices, context=context)
            except Exception as exc:
                if verbose:
                    print(f"Warning: tool final selection failed: {exc}", file=sys.stderr)
                return None

        if attempt + 1 >= attempts:
            break
        error = stderr or "Không có stdout."
        messages = _retry_messages(
            question,
            choices,
            previous_response=last_response,
            previous_code=last_code,
            error=error,
        )

    return None
