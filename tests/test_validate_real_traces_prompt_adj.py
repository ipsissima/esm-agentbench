import numpy as np

from tools import validate_real_traces


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
