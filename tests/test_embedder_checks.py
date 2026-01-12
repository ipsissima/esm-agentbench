import numpy as np
import pytest

from certificates.embedder_checks import DegenerateEmbeddingError, normalize_and_check_embeddings


def test_normalize_and_check_embeddings_ok() -> None:
    np.random.seed(0)
    embs = np.random.randn(6, 4)
    normalized = normalize_and_check_embeddings(embs)
    assert normalized.shape == embs.shape
    row_norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(row_norms, 1.0, atol=1e-6)


def test_normalize_and_check_embeddings_collapsed() -> None:
    embs = np.ones((5, 3))
    with pytest.raises(DegenerateEmbeddingError, match="collapsed"):
        normalize_and_check_embeddings(embs)
