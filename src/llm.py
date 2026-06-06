from __future__ import annotations

from pathlib import Path
from typing import Any

from src.extract import letter_set, validate_letter


_PROMPT_T1 = "Trả lời bằng đúng 1 chữ cái."
_PROMPT_T2 = "Phân tích ngắn từng đáp án rồi chọn đúng 1 chữ cái."
_PROMPT_T3 = (
    "Hãy loại trừ các đáp án sai rõ ràng trước, rồi chọn đúng 1 chữ cái còn lại."
)
_COT_SYSTEM_PROMPT = (
    "Bạn là một trợ lý làm bài trắc nghiệm tiếng Việt. "
    "Hãy suy luận theo đúng 4 bước, ngắn gọn và tập trung:\n"
    "Bước 1: Xác định câu hỏi đang hỏi gì (1 câu ngắn).\n"
    "Bước 2: Đánh giá từng đáp án A, B, C, D... ngắn gọn (1 dòng/đáp án).\n"
    "Bước 3: Loại bỏ phương án sai rõ, ghi nhận phương án còn khả thi.\n"
    "Bước 4: Kết luận với cú pháp 'Đáp án cuối: <letter>'."
)
_COT_CLOSING_INSTRUCTION = (
    "Hãy suy luận theo 4 bước trên rồi kết luận với cú pháp Đáp án cuối: <chữ cái>."
)
_VERIFY_SYSTEM_PROMPT = (
    "Bạn là trợ lý xác minh đáp án trắc nghiệm. Hãy đánh giá lại lựa chọn vừa "
    "đưa ra, so sánh nó với các phương án khác, rồi chỉ trả lời bằng đúng 1 chữ "
    "cái — chữ cái mà bạn tin là đáp án đúng nhất sau khi cân nhắc kỹ."
)
_VERIFY_USER_TEMPLATE = (
    "Câu hỏi:\n{question}\n\n"
    "Lựa chọn:\n{choices_text}\n\n"
    "Đáp án đã chọn: {chosen_letter}\n\n"
    "Hãy kiểm tra lại bằng cách so sánh {chosen_letter} với từng phương án còn lại. "
    "Nếu {chosen_letter} vẫn đúng nhất, trả lời {chosen_letter}. Nếu không, chọn đáp án "
    "khác. Chỉ trả lời 1 chữ cái."
)


class LLMAnswerer:
    def __init__(
        self,
        model_path: Path,
        n_gpu_layers: int = 20,
        n_ctx: int = 8192,
        seed: int = 42,
        verbose: bool = False,
        use_cot: bool = False,
        cot_max_tokens: int = 350,
    ) -> None:
        # Keep llama_cpp optional so non-inference commands and tests can still import.
        from llama_cpp import Llama, LlamaGrammar

        self.seed = seed
        self.use_cot = use_cot
        self.cot_max_tokens = cot_max_tokens
        self._grammar_class = LlamaGrammar
        self._grammars: dict[int, Any] = {}
        self.llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            seed=seed,
            verbose=verbose,
        )

    def grammar_for(self, n_choices: int) -> Any:
        if n_choices not in self._grammars:
            if not 1 <= n_choices <= 26:
                raise ValueError("n_choices must be between 1 and 26")
            letters = letter_set(n_choices)
            grammar = "root ::= " + " | ".join(f'"{letter}"' for letter in letters)
            self._grammars[n_choices] = self._grammar_class.from_string(grammar)
        return self._grammars[n_choices]

    @staticmethod
    def _choice_lines(choices: list[str]) -> str:
        return "\n".join(
            f"{letter}. {choice}"
            for letter, choice in zip(letter_set(len(choices)), choices)
        )

    @staticmethod
    def _many_choice_hint(choices: list[str]) -> str:
        if len(choices) <= 4:
            return ""
        last_letter = letter_set(len(choices))[-1]
        return (
            f"\n\nLưu ý: có {len(choices)} lựa chọn từ A đến {last_letter}. "
            "Hãy xem xét kỹ TẤT CẢ lựa chọn trước khi chọn."
        )

    @staticmethod
    def _retrieved_context(retrieved: list[dict[str, Any]] | None) -> str:
        if not retrieved:
            return ""
        lines = [
            "Ngữ cảnh liên quan từ Wikipedia tiếng Việt "
            "(sắp xếp theo độ liên quan):"
        ]
        for index, record in enumerate(retrieved, start=1):
            title = str(record.get("title", ""))
            text = str(record.get("text", ""))[:1200]
            lines.append(f"[{index}] {title}: {text}")
        return "\n".join(lines) + "\n\n"

    def _user_prompt(
        self,
        question: str,
        choices: list[str],
        context: str | None,
        closing_instruction: str,
        retrieved: list[dict[str, Any]] | None = None,
    ) -> str:
        context_section = f"\n\nNgữ cảnh:\n{context}" if context else ""
        return (
            f"{self._retrieved_context(retrieved)}"
            f"Câu hỏi:\n{question}{context_section}\n\n"
            f"Lựa chọn:\n{self._choice_lines(choices)}"
            f"{self._many_choice_hint(choices)}\n\n"
            f"{closing_instruction}"
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        seed: int | None = None,
        **kwargs: Any,
    ) -> str:
        completion = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=self.seed if seed is None else seed,
            **kwargs,
        )
        content = completion["choices"][0]["message"]["content"]
        return content if isinstance(content, str) else ""

    def answer_mcq(
        self,
        question: str,
        choices: list[str],
        context: str | None = None,
        retrieved: list[dict[str, Any]] | None = None,
        closing_instruction: str = _PROMPT_T1,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý trả lời câu hỏi trắc nghiệm. "
                    "Chỉ trả lời đúng một chữ cái tương ứng với lựa chọn tốt nhất."
                ),
            },
            {
                "role": "user",
                "content": self._user_prompt(
                    question,
                    choices,
                    context,
                    closing_instruction,
                    retrieved=retrieved,
                ),
            },
        ]
        completion = self.llm.create_chat_completion(
            messages=messages,
            grammar=self.grammar_for(len(choices)),
            temperature=0.0,
            max_tokens=2,
            seed=self.seed,
        )
        content = completion["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM completion did not contain an answer")

        answer = validate_letter(content, len(choices))
        if answer != content.strip().upper():
            raise ValueError(f"LLM returned invalid answer {content!r}")
        return answer

    def verify_letter(
        self,
        question: str,
        choices: list[str],
        chosen_letter: str,
        context: str | None = None,
        retrieved: list[dict[str, Any]] | None = None,
    ) -> str:
        chosen = validate_letter(chosen_letter, len(choices))
        prompt = _VERIFY_USER_TEMPLATE.format(
            question=question,
            choices_text=self._choice_lines(choices),
            chosen_letter=chosen,
        )
        if context:
            prompt += f"\n\nNgữ cảnh:\n{context}"
        if retrieved:
            prompt += f"\n\n{self._retrieved_context(retrieved).rstrip()}"
        messages = [
            {"role": "system", "content": _VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        completion = self.llm.create_chat_completion(
            messages=messages,
            grammar=self.grammar_for(len(choices)),
            temperature=0.0,
            max_tokens=2,
            seed=self.seed,
        )
        content = completion["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            return chosen

        answer = validate_letter(content, len(choices), fallback=chosen)
        if answer != content.strip().upper():
            return chosen
        return answer

    def answer_mcq_self_consistent(
        self,
        question: str,
        choices: list[str],
        context: str | None = None,
        retrieved: list[dict[str, Any]] | None = None,
    ) -> str:
        votes = [
            self.answer_mcq(
                question,
                choices,
                context=context,
                retrieved=retrieved,
                closing_instruction=instruction,
            )
            for instruction in (_PROMPT_T1, _PROMPT_T2, _PROMPT_T3)
        ]
        counts: dict[str, int] = {}
        for letter in votes:
            counts[letter] = counts.get(letter, 0) + 1
        for letter, count in counts.items():
            if count >= 2:
                return letter
        return votes[0]

    def _cot_reasoning(
        self,
        question: str,
        choices: list[str],
        context: str | None = None,
        retrieved: list[dict[str, Any]] | None = None,
    ) -> str:
        reasoning_messages = [
            {
                "role": "system",
                "content": _COT_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": self._user_prompt(
                    question,
                    choices,
                    context,
                    _COT_CLOSING_INSTRUCTION,
                    retrieved=retrieved,
                ),
            },
        ]
        return self.complete(
            reasoning_messages,
            max_tokens=self.cot_max_tokens,
            temperature=0.0,
            seed=self.seed,
        )

    def _extract_cot_answer(
        self,
        question: str,
        choices: list[str],
        reasoning: str,
        context: str | None = None,
        retrieved: list[dict[str, Any]] | None = None,
        closing_instruction: str = "Dựa trên suy luận trên, trả lời bằng đúng 1 chữ cái.",
    ) -> str:
        extraction_messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý trả lời câu hỏi trắc nghiệm. "
                    "Chỉ trả lời đúng một chữ cái tương ứng với lựa chọn tốt nhất."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Suy luận:\n{reasoning}\n\n"
                    + self._user_prompt(
                        question,
                        choices,
                        context,
                        closing_instruction,
                        retrieved=retrieved,
                    )
                ),
            },
        ]
        completion = self.llm.create_chat_completion(
            messages=extraction_messages,
            grammar=self.grammar_for(len(choices)),
            temperature=0.0,
            max_tokens=2,
            seed=self.seed,
        )
        content = completion["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM completion did not contain an answer")

        answer = validate_letter(content, len(choices))
        if answer != content.strip().upper():
            raise ValueError(f"LLM returned invalid answer {content!r}")
        return answer

    def answer_mcq_cot(
        self,
        question: str,
        choices: list[str],
        context: str | None = None,
        retrieved: list[dict[str, Any]] | None = None,
    ) -> str:
        reasoning = self._cot_reasoning(
            question,
            choices,
            context=context,
            retrieved=retrieved,
        )
        return self._extract_cot_answer(
            question,
            choices,
            reasoning,
            context=context,
            retrieved=retrieved,
        )

    def answer_mcq_cot_self_consistent(
        self,
        question: str,
        choices: list[str],
        context: str | None = None,
        retrieved: list[dict[str, Any]] | None = None,
    ) -> str:
        reasoning = self._cot_reasoning(
            question,
            choices,
            context=context,
            retrieved=retrieved,
        )
        votes = [
            self._extract_cot_answer(
                question,
                choices,
                reasoning,
                context=context,
                retrieved=retrieved,
                closing_instruction=instruction,
            )
            for instruction in (_PROMPT_T1, _PROMPT_T2, _PROMPT_T3)
        ]
        counts: dict[str, int] = {}
        for letter in votes:
            counts[letter] = counts.get(letter, 0) + 1
        for letter, count in counts.items():
            if count >= 2:
                return letter
        return votes[0]

    def answer(
        self,
        question: str,
        choices: list[str],
        context: str | None = None,
        retrieved: list[dict[str, Any]] | None = None,
    ) -> str:
        if self.use_cot:
            return self.answer_mcq_cot(
                question,
                choices,
                context,
                retrieved=retrieved,
            )
        return self.answer_mcq(
            question,
            choices,
            context,
            retrieved=retrieved,
        )
