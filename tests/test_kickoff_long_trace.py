import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from assessor import kickoff


def test_run_episode_long_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    trace_dir = tmp_path / "demo_traces"
    result = kickoff.run_episode(
        {"prompt": "Solve Fibonacci", "tests": [], "trace_dir": trace_dir},
        max_steps=12,
    )

    assert len(result["trace"]) >= 12
    for entry in result["trace"]:
        assert "timestamp" in entry
        assert "role" in entry
        assert "type" in entry

    trace_path = Path(result["trace_path"])
    assert trace_path.exists()
    with trace_path.open("r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved.get("episode_id") == result["episode_id"]
