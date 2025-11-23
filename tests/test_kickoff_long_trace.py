import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from assessor.kickoff import run_episode  # noqa: E402


def _iso_parse(value: str) -> None:
    datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_trace_schema_and_length(tmp_path: Path):
    task_spec = {
        "prompt": "Compute fibonacci sequence with deterministic agent.",
        "tests": [
            {"name": "fib", "script": "assert fibonacci(7) == 13\nassert fibonacci(0) == 0"}
        ],
        "max_steps": 12,
    }

    result = run_episode(task_spec)

    assert len(result["trace"]) >= 12
    for entry in result["trace"]:
        assert "step" in entry
        assert "role" in entry
        assert "type" in entry
        assert "text" in entry
        assert "timestamp" in entry
        assert "context" in entry
        _iso_parse(str(entry["timestamp"]))

    trace_path = Path(result["trace_path"])
    assert trace_path.exists()
    parsed = json.loads(trace_path.read_text(encoding="utf-8"))
    assert isinstance(parsed, list)
    assert len(parsed) == len(result["trace"])
