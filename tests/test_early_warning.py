import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from assessor import kickoff


def test_pivot_detection_labels_correction(monkeypatch):
    def fake_compute_residuals(_embeddings, threshold):
        residuals = [None, 0.05, 2.0, 0.2, 0.2]
        return residuals, [None] * len(residuals), 2

    monkeypatch.setattr(kickoff, "_compute_residuals", fake_compute_residuals)

    result = kickoff.run_episode({
        "trace_path": "tools/real_traces/sample_gpt4_bad.json",
        "residual_threshold": 0.1,
        "pivot_spike_factor": 5.0,
        "post_stabilize_factor": 0.5,
        "max_steps": 5,
    })
    types = [entry.get("type") for entry in result["trace"]]
    assert "correction" in types
    assert result["early_warning_step"] == 2


def test_good_trace_no_warning(monkeypatch):
    def fake_compute_residuals(_embeddings, threshold):
        residuals = [None, 0.01, 0.02, 0.03, 0.02]
        return residuals, [None] * len(residuals), None

    monkeypatch.setattr(kickoff, "_compute_residuals", fake_compute_residuals)

    result = kickoff.run_episode({
        "trace_path": "tools/real_traces/sample_gpt4_good.json",
        "residual_threshold": 0.5,
        "max_steps": 5,
    })
    types = [entry.get("type") for entry in result["trace"]]
    assert "warning" not in types
    assert result["early_warning_step"] is None
