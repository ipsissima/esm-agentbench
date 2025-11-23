import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assessor import kickoff  # noqa: E402


class DummyModel:
    def __init__(self, raise_on_encode: bool = False):
        self.raise_on_encode = raise_on_encode

    def encode(self, texts, normalize_embeddings: bool = True):
        if self.raise_on_encode:
            raise RuntimeError("boom")
        return np.ones((len(texts), 3))


@pytest.fixture(autouse=True)
def clear_sentence_model_cache(monkeypatch):
    monkeypatch.setattr(kickoff, "_sentence_model_cache", None, raising=False)
    yield
    monkeypatch.setattr(kickoff, "_sentence_model_cache", None, raising=False)


def test_embed_falls_back_when_sentence_transformers_missing(monkeypatch):
    def _fail_model():
        raise ImportError("nope")

    monkeypatch.setattr(kickoff, "_sentence_model", _fail_model)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    arr = kickoff.embed_trace_steps([{"text": "hello"}, {"text": "world"}])
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == 2


def test_embed_uses_tfidf_when_model_none(monkeypatch):
    monkeypatch.setattr(kickoff, "_sentence_model", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    arr = kickoff.embed_trace_steps([{"text": "a"}])
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == 1


def test_embed_handles_model_encode_failure(monkeypatch):
    monkeypatch.setattr(kickoff, "_sentence_model", lambda: DummyModel(raise_on_encode=True))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    arr = kickoff.embed_trace_steps([{"text": "x"}, {"text": "y"}])
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == 2
