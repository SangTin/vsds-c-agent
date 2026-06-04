import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np

from entrypoint import load_config, parse_args
from src.pipeline import answer_question
from src.rag.embedder import BGEEmbedder
from src.rag.retriever import FaissRetriever
from src.schema import Prediction, Question


def test_bge_embedder_encode_query_returns_normalized_float32_vector() -> None:
    flag_embedding = ModuleType("FlagEmbedding")
    model = Mock()
    raw = np.arange(1, 1025, dtype=np.float64)
    model.encode.return_value = {"dense_vecs": raw[None, :]}
    flag_embedding.BGEM3FlagModel = Mock(return_value=model)  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"FlagEmbedding": flag_embedding}):
        embedder = BGEEmbedder()

    vector = embedder.encode_query("Việt Nam")

    flag_embedding.BGEM3FlagModel.assert_called_once_with(  # type: ignore[attr-defined]
        "BAAI/bge-m3",
        device="cpu",
        normalize_embeddings=False,
        use_fp16=False,
    )
    assert vector.shape == (1024,)
    assert vector.dtype == np.float32
    assert np.isclose(np.linalg.norm(vector), 1.0)
    model.encode.assert_called_once_with(
        ["Việt Nam"],
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )


def test_faiss_retriever_returns_metadata_aligned_to_search_indices(
    tmp_path: Path,
) -> None:
    metadata = [
        {"doc_id": "d0", "title": "Zero", "chunk_id": 0, "text": "text zero"},
        {"doc_id": "d1", "title": "One", "chunk_id": 1, "text": "text one"},
        {"doc_id": "d2", "title": "Two", "chunk_id": 2, "text": "text two"},
    ]
    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text(
        "".join(json.dumps(record) + "\n" for record in metadata),
        encoding="utf-8",
    )

    index = Mock()
    index.ntotal = len(metadata)
    index.search.return_value = (
        np.array([[0.9, 0.7]], dtype=np.float32),
        np.array([[2, 0]], dtype=np.int64),
    )
    faiss = ModuleType("faiss")
    faiss.read_index = Mock(return_value=index)  # type: ignore[attr-defined]
    embedder = Mock()
    query_vector = np.ones(1024, dtype=np.float32)
    embedder.encode_query.return_value = query_vector

    with patch.dict(sys.modules, {"faiss": faiss}):
        retriever = FaissRetriever(
            tmp_path / "index.faiss",
            metadata_path,
            embedder,
            top_k=2,
        )

    results = retriever.retrieve("query")

    embedder.encode_query.assert_called_once_with("query")
    searched_vector, searched_k = index.search.call_args.args
    np.testing.assert_array_equal(searched_vector, query_vector[None, :])
    assert searched_k == 2
    assert [result["doc_id"] for result in results] == ["d2", "d0"]
    assert [result["score"] for result in results] == [
        float(np.float32(0.9)),
        float(np.float32(0.7)),
    ]


def test_pipeline_skips_retriever_for_question_with_passage() -> None:
    question = Question(
        "q1",
        "Đoạn thông tin:\nNội dung có sẵn.\nCâu hỏi: Chọn đáp án?",
        ["one", "two"],
    )
    llm = Mock()
    llm.answer.return_value = "B"
    retriever = Mock()

    prediction = answer_question(question, llm=llm, retriever=retriever)

    assert prediction == Prediction("q1", "B")
    retriever.retrieve.assert_not_called()
    llm.answer.assert_called_once_with(
        question.question,
        question.choices,
        retrieved=None,
    )


def test_pipeline_retrieves_once_and_forwards_results_to_llm() -> None:
    question = Question("q1", "Thủ đô Việt Nam là gì?", ["Hà Nội", "Huế"])
    retrieved = [{"title": "Hà Nội", "text": "Hà Nội là thủ đô Việt Nam."}]
    llm = Mock()
    llm.answer.return_value = "A"
    retriever = Mock()
    retriever.retrieve.return_value = retrieved

    prediction = answer_question(question, llm=llm, retriever=retriever)

    assert prediction == Prediction("q1", "A")
    retriever.retrieve.assert_called_once_with(question.question)
    llm.answer.assert_called_once_with(
        question.question,
        question.choices,
        retrieved=retrieved,
    )


def test_pipeline_retriever_exception_falls_back_to_no_context() -> None:
    question = Question("q1", "Question", ["one", "two"])
    llm = Mock()
    llm.answer.return_value = "A"
    retriever = Mock()
    retriever.retrieve.side_effect = RuntimeError("search failed")

    prediction = answer_question(question, llm=llm, retriever=retriever)

    assert prediction == Prediction("q1", "A")
    llm.answer.assert_called_once_with(
        question.question,
        question.choices,
        retrieved=None,
    )


def test_rag_cli_uses_none_sentinel_and_supports_overrides() -> None:
    assert parse_args([]).rag is None
    assert parse_args(["--rag"]).rag is True
    assert parse_args(["--no-rag"]).rag is False
    assert parse_args(["--rag-device", "cuda"]).rag_device == "cuda"


def test_load_config_reads_rag_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.touch()
    yaml = ModuleType("yaml")
    yaml.safe_load = Mock(  # type: ignore[attr-defined]
        return_value={
            "rag": {
                "enabled": True,
                "index_path": "custom/index.faiss",
                "metadata_path": "custom/metadata.jsonl",
                "top_k": 5,
                "device": "cuda",
                "model_name": "custom/model",
            }
        }
    )

    with patch.dict(sys.modules, {"yaml": yaml}):
        rag_config = load_config(config_path)["rag"]

    assert rag_config == {
        "enabled": True,
        "index_path": "custom/index.faiss",
        "metadata_path": "custom/metadata.jsonl",
        "top_k": 5,
        "device": "cuda",
        "model_name": "custom/model",
    }
