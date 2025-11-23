import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath("."))
from assessor import kickoff


def test_embed_trace_steps_tfidf_fallback(monkeypatch):
    """Forces sentence-transformers import failure and checks TF-IDF fallback."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(kickoff, "OPENAI_KEY", None)
    monkeypatch.setattr(kickoff, "openai", None)

    def _fail_sentence_model():
        raise ImportError("forced failure")

    monkeypatch.setattr(kickoff, "_sentence_model", _fail_sentence_model)

    trace = [{"text": "hello world"}, {"text": "another step"}]
    embeddings = kickoff.embed_trace_steps(trace)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(trace)
    assert embeddings.dtype.kind == "f"
    assert np.isfinite(embeddings).all()
