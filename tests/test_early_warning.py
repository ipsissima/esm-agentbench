import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from assessor import kickoff


def test_early_warning_good(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "")

    def fake_embed(trace):
        return [[i * 0.1, 0.0] for i in range(len(trace))]

    def fake_residuals(embeddings, threshold=0.1):
        n = len(embeddings)
        return [None] + [0.01] * (n - 1), [None] * n, None

    monkeypatch.setattr(kickoff, "embed_trace_steps", fake_embed)
    monkeypatch.setattr(kickoff, "_compute_residuals", fake_residuals)

    result = kickoff.run_episode(
        {"prompt": "Compute Fibonacci sequence", "tests": [], "trace_dir": tmp_path},
        max_steps=6,
    )
    assert result["early_warning_step"] is None


def test_early_warning_bad(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "")

    def fake_embed(trace):
        return [[i * 0.1, 0.0] for i in range(len(trace))]

    def fake_residuals(embeddings, threshold=0.1):
        n = len(embeddings)
        residuals = [None, 0.02, 0.15] + [0.02] * max(0, n - 3)
        pred_errors = [None] * n
        return residuals[:n], pred_errors, None

    monkeypatch.setattr(kickoff, "embed_trace_steps", fake_embed)
    monkeypatch.setattr(kickoff, "_compute_residuals", fake_residuals)

    result = kickoff.run_episode(
        {"prompt": "Bad CoT about cakes", "tests": [], "trace_dir": tmp_path},
        max_steps=6,
    )
    assert result["early_warning_step"] == 2 or result["early_warning_step"] == 3
