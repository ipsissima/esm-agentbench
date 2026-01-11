from pathlib import Path
from typing import Any

import numpy as np
import pytest

import sys

# Ensure repo root importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assessor import kickoff  # noqa: E402


@pytest.fixture(autouse=True)
def clear_sentence_model_cache(monkeypatch):
    monkeypatch.setattr(kickoff, "_sentence_model_cache", None, raising=False)
    yield
    monkeypatch.setattr(kickoff, "_sentence_model_cache", None, raising=False)


def _disable_openai(monkeypatch):
    monkeypatch.setattr(kickoff, "OPENAI_KEY", None, raising=False)
    monkeypatch.setattr(kickoff, "openai", None, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _assert_embedding_shape(arr: np.ndarray, rows: int) -> None:
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == rows
    assert arr.dtype == float


def test_embed_raises_when_sentence_model_import_errors(monkeypatch):
    """SBERT import failure should raise RuntimeError (no silent fallback)."""
    _disable_openai(monkeypatch)
    monkeypatch.delenv("ESM_FORCE_TFIDF", raising=False)

    def _boom() -> Any:
        # Simulate what real _sentence_model does when import fails
        raise RuntimeError(
            "sentence-transformers required but unavailable: missing. "
            "Install with: pip install sentence-transformers. "
            "To force TF-IDF fallback (not recommended), set ESM_FORCE_TFIDF=1"
        )

    monkeypatch.setattr(kickoff, "_sentence_model", _boom)

    with pytest.raises(RuntimeError, match="sentence-transformers required"):
        kickoff.embed_trace_steps([{"text": "hello"}, {"text": "world"}])


def test_embed_raises_when_model_encode_fails(monkeypatch):
    """SBERT encode failure should raise RuntimeError (no silent fallback)."""
    _disable_openai(monkeypatch)
    monkeypatch.delenv("ESM_FORCE_TFIDF", raising=False)

    class DummyModel:
        def encode(self, texts, normalize_embeddings: bool = True):
            raise RuntimeError("boom")

    monkeypatch.setattr(kickoff, "_sentence_model", lambda: DummyModel())

    with pytest.raises(RuntimeError, match="SBERT encoding failed"):
        kickoff.embed_trace_steps([{"text": "x"}, {"text": "y"}])


def test_embed_uses_tfidf_when_forced(monkeypatch):
    """ESM_FORCE_TFIDF=1 should use TF-IDF directly (for debugging)."""
    _disable_openai(monkeypatch)
    monkeypatch.setenv("ESM_FORCE_TFIDF", "1")

    # Even if _sentence_model would fail, TF-IDF is used directly
    def _boom() -> Any:
        raise ImportError("should not be called")

    monkeypatch.setattr(kickoff, "_sentence_model", _boom)

    arr = kickoff.embed_trace_steps([{"text": "hello"}, {"text": "world"}])
    _assert_embedding_shape(arr, 2)


def test_embed_uses_sbert_when_available(monkeypatch):
    """SBERT should be used when available (normal operation)."""
    _disable_openai(monkeypatch)
    monkeypatch.delenv("ESM_FORCE_TFIDF", raising=False)

    class DummyModel:
        def encode(self, texts, normalize_embeddings: bool = True):
            # Return 384-dim embeddings like real SBERT
            return np.random.randn(len(texts), 384).astype(float)

    monkeypatch.setattr(kickoff, "_sentence_model", lambda: DummyModel())

    arr = kickoff.embed_trace_steps([{"text": "hello"}, {"text": "world"}])
    _assert_embedding_shape(arr, 2)
    assert arr.shape[1] == 384  # SBERT dimension
