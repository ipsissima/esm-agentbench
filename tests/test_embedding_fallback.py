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


def test_embed_falls_back_when_sentence_model_import_errors(monkeypatch):
    _disable_openai(monkeypatch)

    def _boom() -> Any:
        raise ImportError("missing")

    monkeypatch.setattr(kickoff, "_sentence_model", _boom)
    arr = kickoff.embed_trace_steps([{"text": "hello"}, {"text": "world"}])
    _assert_embedding_shape(arr, 2)


def test_embed_uses_tfidf_when_model_none(monkeypatch):
    _disable_openai(monkeypatch)
    monkeypatch.setattr(kickoff, "_sentence_model", lambda: None)
    arr = kickoff.embed_trace_steps([{"text": "a"}])
    _assert_embedding_shape(arr, 1)


def test_embed_handles_model_encode_failure(monkeypatch):
    _disable_openai(monkeypatch)

    class DummyModel:
        def encode(self, texts, normalize_embeddings: bool = True):
            raise RuntimeError("boom")

    monkeypatch.setattr(kickoff, "_sentence_model", lambda: DummyModel())
    arr = kickoff.embed_trace_steps([{"text": "x"}, {"text": "y"}])
    _assert_embedding_shape(arr, 2)
