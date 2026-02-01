"""Embedding port definitions."""
from __future__ import annotations

from typing import List, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class EmbedderPort(Protocol):
    """Port for embedding implementations."""

    def load(self) -> None:
        """Load or initialize the embedding model."""

    def unload(self) -> None:
        """Unload the embedding model and free resources."""

    def embed(self, text: str) -> np.ndarray:
        """Embed a single piece of text."""

    def embed_batch(self, texts: Sequence[str], batch_size: int = 32) -> List[np.ndarray]:
        """Embed a batch of text inputs."""

    def get_embedder_id(self) -> str:
        """Return a stable embedder identifier for provenance."""
