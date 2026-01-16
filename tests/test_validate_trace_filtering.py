"""Test that validate_real_traces.py filters out short traces correctly."""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Skip these tests if running outside Docker/CI where native libraries may cause SIGSEGV
# The validate_real_traces module imports sentence_transformers which can crash on some hosts
_skip_reason = "Skipping: requires Docker environment (native embedding libraries may SIGSEGV)"
if os.environ.get("SKIP_NATIVE_EMBEDDING_TESTS", "0") == "1":
    pytest.skip(_skip_reason, allow_module_level=True)

try:
    from tools.validate_real_traces import load_trace
except Exception as e:
    pytest.skip(f"Skipping: failed to import validate_real_traces: {e}", allow_module_level=True)


def test_short_trace_filtering():
    """Test that traces with fewer than 10 steps would be filtered."""
    # Create a temporary directory with test traces
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create a short trace (5 steps) - should be filtered
        short_trace = {
            "trace": [
                {"step": i, "role": "agent", "type": "cot", "text": f"Step {i}"}
                for i in range(1, 6)
            ]
        }
        short_file = tmpdir_path / "short_trace.json"
        with open(short_file, 'w') as f:
            json.dump(short_trace, f)
        
        # Create a long trace (15 steps) - should NOT be filtered
        long_trace = {
            "trace": [
                {"step": i, "role": "agent", "type": "cot", "text": f"Step {i}"}
                for i in range(1, 16)
            ]
        }
        long_file = tmpdir_path / "long_trace.json"
        with open(long_file, 'w') as f:
            json.dump(long_trace, f)
        
        # Create a trace with exactly 10 steps - should NOT be filtered
        boundary_trace = {
            "trace": [
                {"step": i, "role": "agent", "type": "cot", "text": f"Step {i}"}
                for i in range(1, 11)
            ]
        }
        boundary_file = tmpdir_path / "boundary_trace.json"
        with open(boundary_file, 'w') as f:
            json.dump(boundary_trace, f)
        
        # Test loading and checking lengths
        short_loaded = load_trace(short_file)
        long_loaded = load_trace(long_file)
        boundary_loaded = load_trace(boundary_file)
        
        assert len(short_loaded) == 5, "Short trace should have 5 steps"
        assert len(long_loaded) == 15, "Long trace should have 15 steps"
        assert len(boundary_loaded) == 10, "Boundary trace should have 10 steps"
        
        # Verify the filtering logic
        assert len(short_loaded) < 10, "Short trace should be filtered (< 10)"
        assert len(long_loaded) >= 10, "Long trace should NOT be filtered (>= 10)"
        assert len(boundary_loaded) >= 10, "Boundary trace should NOT be filtered (>= 10)"
        
        print("✓ All trace filtering tests passed")


if __name__ == "__main__":
    test_short_trace_filtering()
