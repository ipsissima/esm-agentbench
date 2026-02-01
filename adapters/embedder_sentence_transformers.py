"""SentenceTransformers embedder adapter."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ports.embedder import EmbedderPort


class SentenceTransformersEmbedder(EmbedderPort):
    """Adapter for sentence-transformers models."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    def load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for this adapter. "
                "Install with: pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(self._model_name)

    def unload(self) -> None:
        self._model = None

    def embed(self, text: str) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Embedder not loaded. Call load() first.")
        return np.asarray(self._model.encode(text))

    def embed_batch(self, texts: Sequence[str], batch_size: int = 32) -> List[np.ndarray]:
        if self._model is None:
            raise RuntimeError("Embedder not loaded. Call load() first.")
        embeddings = self._model.encode(list(texts), batch_size=batch_size)
        return [np.asarray(vec) for vec in embeddings]

    def get_embedder_id(self) -> str:
        return self._model_name
