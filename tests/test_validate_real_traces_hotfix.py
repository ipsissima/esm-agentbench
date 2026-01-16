"""Tests for validate_real_traces hotfix filtering and starvation handling."""
import json
import os
from pathlib import Path

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


def _write_trace(path: Path, steps: int = 10) -> None:
    trace = {
        "trace": [
            {"step": i, "role": "agent", "type": "cot", "text": f"Step {i}"}
            for i in range(1, steps + 1)
        ]
    }
    path.write_text(json.dumps(trace))


def test_sentinel_detection_skips(tmp_path, monkeypatch):
    sentinel_file = tmp_path / "sentinel_trace.json"
    coherent_file = tmp_path / "coherent_trace.json"
    _write_trace(sentinel_file)
    _write_trace(coherent_file)

    def fake_analyze(trace, label, meta=None, gamma=None):
        if "sentinel" in label:
            return {
                "label": label,
                "n_steps": len(trace),
                "theoretical_bound": 3.0,
                "residual": 1.0,
                "oos_residual": 1.0,
                "insample_residual": 1.0,
                "r_eff": 1.0,
                "pca_explained": 0.0,
                "pca_tail_estimate": 1.0,
            }
        return {
            "label": label,
            "n_steps": len(trace),
            "theoretical_bound": 0.4,
            "residual": 0.2,
            "oos_residual": 0.2,
            "insample_residual": 0.2,
            "r_eff": 4.0,
            "pca_explained": 0.8,
            "pca_tail_estimate": 0.1,
        }

    monkeypatch.setattr(validate_real_traces, "analyze_trace", fake_analyze)

    result = validate_real_traces.run_real_trace_validation(real_traces_dir=tmp_path)

    assert isinstance(result, dict)
    assert "coherent_trace" in result["results"]
    excluded = {item["label"]: item for item in result["excluded_traces"]}
    assert "sentinel_trace" in excluded
    assert excluded["sentinel_trace"]["reason"] == "sentinel_empty_certificate"
    assert result["excluded_trace_counts"]["sentinel_empty_certificate"] == 1


def test_starvation_rank_test_and_exclusion(tmp_path, monkeypatch, capsys):
    coherent_file = tmp_path / "coherent_trace.json"
    starvation_file = tmp_path / "starvation_trace.json"
    _write_trace(coherent_file)
    _write_trace(starvation_file)

    def fake_analyze(trace, label, meta=None, gamma=None):
        # r_rel is required for the starvation vs coherent rank test
        return {
            "label": label,
            "n_steps": len(trace),
            "theoretical_bound": 0.3 if "coherent" in label else 0.6,
            "residual": 0.2,
            "oos_residual": 0.2,
            "insample_residual": 0.2,
            "r_eff": 6.0 if "coherent" in label else 1.0,
            "r_rel": 0.6 if "coherent" in label else 0.1,  # normalized rank for rank test
            "pca_explained": 0.7,
            "pca_tail_estimate": 0.1,
        }

    monkeypatch.setattr(validate_real_traces, "analyze_trace", fake_analyze)

    result = validate_real_traces.run_real_trace_validation(real_traces_dir=tmp_path)

    assert isinstance(result, dict)
    captured = capsys.readouterr().out
    assert "[Starvation vs Coherent: Rank Test]" in captured
    assert "Starvation mean normalized rank" in captured
    assert "All Good vs All Bad (excluding starvation; starvation tested separately)" in captured
    assert "Need both good and bad (non-starvation) traces for this test" in captured
    assert np.isclose(np.mean([1.0]), np.mean([result["results"]["starvation_trace"]["r_eff"]]))
