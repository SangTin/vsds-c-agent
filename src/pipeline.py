from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from src.agent.loop import solve_with_tools, solve_with_tools_ranking
from src.extract import validate_letter
from src.llm import LLMAnswerer
from src.rag.retriever import FaissRetriever
from src.router import (
    detect_alignment_bait,
    is_law_question,
    is_math_question,
    is_polysci_question,
)
from src.schema import Prediction, Question


_PASSAGE_MARKER = "\u0110o\u1ea1n th\u00f4ng tin:"


def _select_mcq_method(
    llm: LLMAnswerer,
    self_consistency: bool,
) -> Callable[..., str]:
    use_cot = getattr(llm, "use_cot", False) is True
    if use_cot and self_consistency:
        return llm.answer_mcq_cot_self_consistent
    if use_cot:
        return llm.answer_mcq_cot
    if self_consistency:
        return llm.answer_mcq_self_consistent
    return llm.answer_mcq


def _select_non_cot_mcq_method(
    llm: LLMAnswerer,
    self_consistency: bool,
) -> Callable[..., str]:
    if self_consistency:
        return llm.answer_mcq_self_consistent
    return llm.answer_mcq


def _select_mcq_method_for_question(
    llm: LLMAnswerer,
    self_consistency: bool,
    q: Question,
    use_cot_passage: bool,
    cot_passage_max_chars: int,
) -> Callable[..., str]:
    use_cot = getattr(llm, "use_cot", False) is True
    if _PASSAGE_MARKER not in q.question or not use_cot:
        return _select_mcq_method(llm, self_consistency)

    if use_cot_passage and len(q.question) <= cot_passage_max_chars:
        return _select_mcq_method(llm, self_consistency)

    if use_cot_passage:
        print(
            f"Warning: skipping CoT for passage question {q.qid!r}; "
            f"question length {len(q.question)} exceeds "
            f"cot_passage_max_chars={cot_passage_max_chars}",
            file=sys.stderr,
        )
    return _select_non_cot_mcq_method(llm, self_consistency)


def _maybe_verify_letter(
    q: Question,
    fallback_letter: str,
    llm: LLMAnswerer,
    answer: str,
    use_self_verify: bool,
    self_consistency: bool,
    context: str | None = None,
    retrieved: list[dict[str, Any]] | None = None,
) -> str:
    validated = validate_letter(answer, len(q.choices), fallback_letter)
    if not use_self_verify or self_consistency:
        return validated

    verify_letter = getattr(llm, "verify_letter", None)
    if not callable(verify_letter):
        return validated

    try:
        final = verify_letter(
            q.question,
            q.choices,
            validated,
            context=context,
            retrieved=retrieved,
        )
        return validate_letter(final, len(q.choices), fallback=validated)
    except Exception as exc:
        print(
            f"Warning: self-verification failed for {q.qid!r}; "
            f"keeping first-pass answer {validated}: {exc}",
            file=sys.stderr,
        )
        return validated


def _direct_answer(
    q: Question,
    fallback_letter: str,
    llm: LLMAnswerer | None,
    retrieved: list[dict[str, Any]] | None = None,
    self_consistency: bool = False,
    use_cot_passage: bool = False,
    cot_passage_max_chars: int = 3500,
    use_self_verify: bool = False,
) -> Prediction:
    if llm is None:
        return Prediction(q.qid, fallback_letter)

    try:
        if (
            self_consistency
            or getattr(llm, "use_cot", False) is True
            or use_self_verify
        ):
            answer = _select_mcq_method_for_question(
                llm,
                self_consistency,
                q,
                use_cot_passage,
                cot_passage_max_chars,
            )(
                q.question,
                q.choices,
                context=None,
                retrieved=retrieved,
            )
        else:
            answer = llm.answer(q.question, q.choices, retrieved=retrieved)
        validated = _maybe_verify_letter(
            q,
            fallback_letter,
            llm,
            answer,
            use_self_verify,
            self_consistency,
            retrieved=retrieved,
        )
        return Prediction(q.qid, validated)
    except Exception as exc:
        print(
            f"Warning: LLM failed for {q.qid!r}; falling back to "
            f"{fallback_letter}: {exc}",
            file=sys.stderr,
        )
        return Prediction(q.qid, fallback_letter)


def _format_context(
    retrieved: list[dict[str, Any]],
    heading: str,
    default_title: str,
) -> str:
    lines = [heading]
    for index, record in enumerate(retrieved, start=1):
        title = str(record.get("title", "")).strip()
        if not title:
            title = str(record.get("doc_id", default_title))
        text = str(record.get("text", ""))[:1500]
        lines.append(f"[{index}] {title}: {text}")
    return "\n".join(lines)


def _legal_context(retrieved: list[dict[str, Any]]) -> str:
    return _format_context(
        retrieved,
        "Trích văn bản pháp luật liên quan:",
        "Văn bản pháp luật",
    )


def _polysci_context(retrieved: list[dict[str, Any]]) -> str:
    return _format_context(
        retrieved,
        "Trích giáo trình lý luận chính trị:",
        "Giáo trình lý luận chính trị",
    )


def answer_question(
    q: Question,
    fallback: str = "A",
    llm: LLMAnswerer | None = None,
    retriever: FaissRetriever | None = None,
    use_tools: bool = False,
    run_python: Callable[[str], dict[str, Any]] | None = None,
    legal_retriever: FaissRetriever | None = None,
    legal_min_score: float = 0.0,
    polysci_retriever: FaissRetriever | None = None,
    polysci_min_score: float = 0.0,
    self_consistency: bool = False,
    use_pot_ranking: bool = False,
    use_cot_passage: bool = False,
    cot_passage_max_chars: int = 3500,
    use_self_verify: bool = False,
    use_alignment_override: bool = False,
) -> Prediction:
    """Answer one question with the LLM, falling back without breaking the batch."""
    fallback_letter = validate_letter(fallback, len(q.choices))
    if use_alignment_override:
        refusal_letter = detect_alignment_bait(q.question, q.choices)
        if refusal_letter:
            return Prediction(
                q.qid,
                validate_letter(refusal_letter, len(q.choices), fallback_letter),
            )

    if (
        use_tools
        and llm is not None
        and run_python is not None
        and is_math_question(q.question, q.choices)
    ):
        try:
            if use_pot_ranking:
                letter = solve_with_tools_ranking(
                    llm,
                    q.question,
                    q.choices,
                    run_python=run_python,
                )
                if letter:
                    normalized_letter = letter.strip().upper()
                    validated_letter = validate_letter(
                        normalized_letter,
                        len(q.choices),
                        fallback="",
                    )
                    if validated_letter != normalized_letter:
                        letter = None
                if letter:
                    return Prediction(
                        q.qid,
                        normalized_letter,
                    )
            letter = solve_with_tools(llm, q.question, q.choices, run_python)
            if letter:
                return Prediction(
                    q.qid,
                    validate_letter(letter, len(q.choices), fallback_letter),
                )
        except Exception as exc:
            print(
                f"Warning: tool path failed for {q.qid!r}; "
                f"falling back to direct answer: {exc}",
                file=sys.stderr,
            )

    if llm is not None and legal_retriever is not None and is_law_question(
        q.question,
        q.choices,
    ):
        try:
            legal_chunks = legal_retriever.retrieve(q.question)
            top_score = (
                float(legal_chunks[0].get("score", 0.0)) if legal_chunks else 0.0
            )
            if legal_chunks and top_score >= legal_min_score:
                context = _legal_context(legal_chunks)
                answer = _select_mcq_method_for_question(
                    llm,
                    self_consistency,
                    q,
                    use_cot_passage,
                    cot_passage_max_chars,
                )(
                    q.question,
                    q.choices,
                    context=context,
                )
                validated = _maybe_verify_letter(
                    q,
                    fallback_letter,
                    llm,
                    answer,
                    use_self_verify,
                    self_consistency,
                    context=context,
                )
                return Prediction(q.qid, validated)
        except Exception as exc:
            print(
                f"Warning: legal RAG path failed for {q.qid!r}; "
                f"falling back to direct answer: {exc}",
                file=sys.stderr,
            )
        return _direct_answer(
            q,
            fallback_letter,
            llm,
            self_consistency=self_consistency,
            use_cot_passage=use_cot_passage,
            cot_passage_max_chars=cot_passage_max_chars,
            use_self_verify=use_self_verify,
        )

    if llm is not None and polysci_retriever is not None and is_polysci_question(
        q.question,
        q.choices,
    ):
        try:
            polysci_chunks = polysci_retriever.retrieve(q.question)
            top_score = (
                float(polysci_chunks[0].get("score", 0.0))
                if polysci_chunks
                else 0.0
            )
            if polysci_chunks and top_score >= polysci_min_score:
                context = _polysci_context(polysci_chunks)
                answer = _select_mcq_method_for_question(
                    llm,
                    self_consistency,
                    q,
                    use_cot_passage,
                    cot_passage_max_chars,
                )(
                    q.question,
                    q.choices,
                    context=context,
                )
                validated = _maybe_verify_letter(
                    q,
                    fallback_letter,
                    llm,
                    answer,
                    use_self_verify,
                    self_consistency,
                    context=context,
                )
                return Prediction(q.qid, validated)
        except Exception as exc:
            print(
                f"Warning: polysci RAG path failed for {q.qid!r}; "
                f"falling back to direct answer: {exc}",
                file=sys.stderr,
            )
        return _direct_answer(
            q,
            fallback_letter,
            llm,
            self_consistency=self_consistency,
            use_cot_passage=use_cot_passage,
            cot_passage_max_chars=cot_passage_max_chars,
            use_self_verify=use_self_verify,
        )

    retrieved: list[dict[str, Any]] | None = None
    should_retrieve = retriever is not None and _PASSAGE_MARKER not in q.question
    if should_retrieve:
        try:
            retrieved = retriever.retrieve(q.question)
        except Exception as exc:
            print(
                f"Warning: RAG retrieval failed for {q.qid!r}; "
                f"continuing without retrieved context: {exc}",
                file=sys.stderr,
            )

    return _direct_answer(
        q,
        fallback_letter,
        llm,
        retrieved=retrieved,
        self_consistency=self_consistency,
        use_cot_passage=use_cot_passage,
        cot_passage_max_chars=cot_passage_max_chars,
        use_self_verify=use_self_verify,
    )
