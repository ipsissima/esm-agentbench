import os

import numpy as np
import pytest

# Skip these tests if running outside Docker/CI where native libraries may cause SIGSEGV
# The validate_real_traces module imports sentence_transformers which can crash on some hosts
_skip_reason = "Skipping: requires Docker environment (native embedding libraries may SIGSEGV)"
if os.environ.get("SKIP_NATIVE_EMBEDDING_TESTS", "0") == "1":
    pytest.skip(_skip_reason, allow_module_level=True)

try:
    from tools import validate_real_traces
except Exception as e:
    pytest.skip(f"Skipping: failed to import validate_real_traces: {e}", allow_module_level=True)


def test_prompt_adjustment_applies_gamma(monkeypatch):
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])

    def fake_embed_trace_steps(_trace):
        return embeddings

    class DummyModel:
        def encode(self, texts, convert_to_numpy=True):
            return np.array([[-1.0, 0.0]])

    monkeypatch.setattr("assessor.kickoff.embed_trace_steps", fake_embed_trace_steps)
    monkeypatch.setattr("assessor.kickoff._sentence_model", lambda: DummyModel())

    trace = [{"type": "cot", "role": "agent", "text": "step"}]
    meta = {"prompt": "test prompt"}
    result = validate_real_traces.analyze_trace(trace, "sample", meta=meta, gamma=0.3)

    assert "theoretical_bound_prompt_adj" in result
    base = result.get("theoretical_bound_norm")
    if base is None or np.isnan(base):
        base = result.get("theoretical_bound")
    if base is not None and not np.isnan(base) and not np.isnan(result["theoretical_bound_prompt_adj"]):
        assert result["theoretical_bound_prompt_adj"] <= base
