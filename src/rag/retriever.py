from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.rag.embedder import BGEEmbedder


class FaissRetriever:
    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        embedder: BGEEmbedder,
        top_k: int = 3,
        verbose: bool = False,
    ) -> None:
        # Keep FAISS optional for direct-LLM and stub runs.
        import faiss

        self.index: Any = faiss.read_index(str(index_path))
        self.embedder = embedder
        self.top_k = top_k
        self.verbose = verbose
        with metadata_path.open("r", encoding="utf-8") as file:
            self.metadata: list[dict[str, Any]] = [
                json.loads(line) for line in file if line.strip()
            ]

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "FAISS index and metadata row counts differ: "
                f"{self.index.ntotal} != {len(self.metadata)}"
            )

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        query_vector = np.ascontiguousarray(
            self.embedder.encode_query(query)[None, :],
            dtype=np.float32,
        )
        scores, indices = self.index.search(query_vector, self.top_k)
        results: list[dict[str, Any]] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            record = dict(self.metadata[int(index)])
            record["score"] = float(score)
            results.append(record)
        return results
