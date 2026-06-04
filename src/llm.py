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
    ) -> None:
        # Keep llama_cpp optional so non-inference commands and tests can still import.
        from llama_cpp import Llama, LlamaGrammar

        self.seed = seed
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

    def answer_mcq(
        self,
        question: str,
        choices: list[str],
        context: str | None = None,
    ) -> str:
        choice_lines = "\n".join(
            f"{letter}. {choice}"
            for letter, choice in zip(letter_set(len(choices)), choices)
        )
        context_section = f"\n\nNgữ cảnh:\n{context}" if context else ""
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
                "content": f"Câu hỏi:\n{question}{context_section}\n\nLựa chọn:\n{choice_lines}",
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
