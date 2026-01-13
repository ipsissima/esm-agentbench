"""Tests for submission trace attestation metadata."""
import json
from pathlib import Path

import pytest


def test_submission_index_attestation():
    submissions_dir = Path(__file__).parent.parent / "submissions"
    if not submissions_dir.exists():
        pytest.skip("submissions directory not present")

    index_files = list(submissions_dir.glob("**/experiment_traces_real_hf/index.json"))
    if not index_files:
        pytest.skip("no submission indexes available")

    for index_file in index_files:
        with open(index_file) as f:
            data = json.load(f)

        assert data.get("run_generator"), f"Missing run_generator in {index_file}"
        assert "run_real_agents.py" in data["run_generator"]

        attestation = data.get("attestation")
        assert attestation, f"Missing attestation in {index_file}"
        assert attestation.get("run_signature"), f"Missing run_signature in {index_file}"
        assert isinstance(attestation.get("trace_hashes"), list)
