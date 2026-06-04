from pathlib import Path
from typing import Any

from src.extract import letter_set, validate_letter


class LLMAnswerer:
    def __init__(
        self,
        model_path: Path,
        n_gpu_layers: int = 20,
        n_ctx: int = 8192,
        seed: int = 42,
        verbose: bool = False,
        use_cot: bool = False,
        cot_max_tokens: int = 200,
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

    def _user_prompt(
        self,
        question: str,
        choices: list[str],
        context: str | None,
        closing_instruction: str,
    ) -> str:
        context_section = f"\n\nNgữ cảnh:\n{context}" if context else ""
        return (
            f"Câu hỏi:\n{question}{context_section}\n\n"
            f"Lựa chọn:\n{self._choice_lines(choices)}"
            f"{self._many_choice_hint(choices)}\n\n"
            f"{closing_instruction}"
        )

    def answer_mcq(
        self,
        question: str,
        choices: list[str],
        context: str | None = None,
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
                    "Trả lời bằng đúng 1 chữ cái.",
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

    def answer_mcq_cot(
        self,
        question: str,
        choices: list[str],
        context: str | None = None,
    ) -> str:
        reasoning_messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là một trợ lý làm bài trắc nghiệm. Hãy suy luận ngắn gọn "
                    "(tối đa 5 câu) rồi nêu đáp án cuối cùng."
                ),
            },
            {
                "role": "user",
                "content": self._user_prompt(
                    question,
                    choices,
                    context,
                    "Hãy suy luận ngắn gọn từng bước rồi kết luận đáp án bằng cú pháp "
                    "'Đáp án cuối: <letter>'.",
                ),
            },
        ]
        reasoning_completion = self.llm.create_chat_completion(
            messages=reasoning_messages,
            temperature=0.0,
            max_tokens=self.cot_max_tokens,
            seed=self.seed,
        )
        reasoning_content = reasoning_completion["choices"][0]["message"]["content"]
        reasoning = reasoning_content if isinstance(reasoning_content, str) else ""

        context_section = f"\n\nNgữ cảnh:\n{context}" if context else ""
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
                    f"Câu hỏi:\n{question}{context_section}\n\n"
                    f"Các lựa chọn:\n{self._choice_lines(choices)}"
                    f"{self._many_choice_hint(choices)}\n\n"
                    "Dựa trên suy luận trên, trả lời bằng đúng 1 chữ cái."
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

    def answer(
        self,
        question: str,
        choices: list[str],
        context: str | None = None,
    ) -> str:
        if self.use_cot:
            return self.answer_mcq_cot(question, choices, context)
        return self.answer_mcq(question, choices, context)
