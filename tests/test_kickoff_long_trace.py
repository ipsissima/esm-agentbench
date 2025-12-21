import os
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from assessor import kickoff


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)
def test_long_trace_from_cached_real():
    result = kickoff.run_episode({
        "trace_path": "tools/real_traces/sample_gpt4_good.json",
        "max_steps": 12,
        "use_cached_real": True,
    })
    trace = result["trace"]
    assert len(trace) >= 12
    for entry in trace:
        assert "step" in entry
        assert "role" in entry
        assert "type" in entry
        assert "text" in entry
    assert result["per_step_residuals"]
    assert result["early_warning_step"] is None or isinstance(result["early_warning_step"], int)
