import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assessor import kickoff
from certificates.make_certificate import compute_certificate


def test_run_episode_produces_long_trace():
    result = kickoff.run_episode({"prompt": "demo long trace", "tests": []}, max_steps=12)
    trace = result["trace"]

    assert len(trace) >= 12
    for entry in trace:
        for key in ["step", "role", "type", "text", "timestamp", "context"]:
            assert key in entry

    trace_path = Path("demo_traces") / f"{result['episode_id']}.json"
    assert trace_path.exists()
    with trace_path.open("r", encoding="utf-8") as f:
        saved = json.load(f)
    assert isinstance(saved, list)


def test_chain_of_thought_certificate_runs():
    result = kickoff.run_episode({"prompt": "certificate check", "tests": []}, max_steps=12)
    embeddings = kickoff.embed_trace_steps(result["trace"])
    cert = compute_certificate(embeddings)

    assert "theoretical_bound" in cert
