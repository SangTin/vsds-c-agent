from __future__ import annotations

from typing import Any

import numpy as np


class BGEEmbedder:
    DIMENSION = 1024

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        max_length: int = 512,
        verbose: bool = False,
    ) -> None:
        # Keep FlagEmbedding optional for direct-LLM and stub runs.
        from FlagEmbedding import BGEM3FlagModel

        self.model: Any = BGEM3FlagModel(
            model_name,
            device=device,
            normalize_embeddings=False,
            use_fp16=device != "cpu",
        )
        # Default BGE-m3 max_length is 8192; corpus chunks are ~600 tokens so
        # padding to 8192 wastes ~14x compute. 512 is enough for our chunks.
        self.max_length = max_length
        self.verbose = verbose

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("BGE-m3 returned a zero-length embedding")
        return np.ascontiguousarray(vectors / norms, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        result = self.model.encode(
            [text],
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = self._normalize(result["dense_vecs"])
        if dense.shape != (1, self.DIMENSION):
            raise ValueError(
                f"Expected one {self.DIMENSION}-dimensional query vector, "
                f"got {dense.shape}"
            )
        return dense[0]

    def encode_corpus(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.DIMENSION), dtype=np.float32)
        result = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = self._normalize(result["dense_vecs"])
        if dense.shape != (len(texts), self.DIMENSION):
            raise ValueError(
                f"Expected {(len(texts), self.DIMENSION)} corpus vectors, "
                f"got {dense.shape}"
            )
        return dense
