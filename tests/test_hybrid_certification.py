"""Integration tests for Phase 3: Hybrid Certification (Execution Grounding).

This test suite specifically targets the "Stable-but-Wrong" failure mode:
An agent produces reasoning that is perfectly stable (low spectral bound) but
produces code that is functionally broken (fails all tests).

Phase 3 ensures that such agents are NOT certified, despite their "stable" reasoning.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pytest

from assessor.kickoff import _extract_verify_block, run_agent_verify_block
from certificates.make_certificate import certify_episode, compute_certificate
from esmassessor.artifact_schema import CertifiedVerdict


class TestVerifyBlockExtraction:
    """Test extraction of agent-generated verify() blocks."""

    def test_extract_verify_block_simple(self):
        """Test extraction of a simple verify block."""
        response = """
Here's my solution:

```python
def fibonacci(n):
    return 0  # Always wrong!
```

def verify():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(5) == 5
"""
        verify_block = _extract_verify_block(response)
        assert verify_block is not None
        assert "def verify()" in verify_block
        assert "assert fibonacci(0) == 0" in verify_block

    def test_extract_verify_block_not_present(self):
        """Test that None is returned if no verify block exists."""
        response = "Here's my solution without a verify block."
        verify_block = _extract_verify_block(response)
        assert verify_block is None

    def test_extract_verify_block_with_code(self):
        """Test extraction with mixed code blocks."""
        response = """
```python
def fibonacci(n):
    if n < 2: return n
    return fibonacci(n-1) + fibonacci(n-2)
```

def verify():
    assert fibonacci(5) == 5
    print("All tests passed!")
"""
        verify_block = _extract_verify_block(response)
        assert verify_block is not None
        assert "def verify()" in verify_block


class TestStableButWrong:
    """Test the core "Stable-but-Wrong" detection scenario.

    This is the critical use case: an agent produces consistent but incorrect code.
    The spectral analysis should show LOW drift (stable embeddings), but the
    execution should FAIL (tests don't pass). Phase 3 must catch this.
    """

    def test_stable_but_wrong_always_returns_zero(self):
        """Force an agent to produce code that always returns 0.

        Scenario:
        - Embeddings show very low residuals (stable, consistent trajectory)
        - But the code is completely wrong (returns 0 for fibonacci(5) instead of 5)
        - Spectral bound should be LOW (stable)
        - Execution should FAIL (tests fail)
        - Certificate should be FAIL_EXECUTION (not PASS)
        """
        # Create a synthetic trace that shows stability (low residuals)
        # but the code is broken
        agent_response = """
Step 1: I understand the fibonacci problem.
Step 2: I'll implement a simple solution.
Step 3: Here's my code:

```python
def fibonacci(n):
    return 0  # Perfectly consistent but wrong!
```

def verify():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 0  # Wrong expectation
    assert fibonacci(5) == 0  # Wrong!
"""

        # Simulate embeddings with low variance (stable)
        # Use repeated similar embeddings to create low-residual trajectory
        stable_embeddings = [
            [0.1, 0.2, 0.3],
            [0.11, 0.21, 0.31],  # Very close to first
            [0.12, 0.22, 0.32],  # Consistent drift
            [0.13, 0.23, 0.33],  # Stable progression
        ]

        spectral_cert = compute_certificate(stable_embeddings)

        # The spectral certificate should show LOW bound (stable)
        theoretical_bound = spectral_cert.get("theoretical_bound", float("inf"))
        assert theoretical_bound < 1.0, (
            f"Expected stable embeddings to have low bound, got {theoretical_bound}"
        )

        # But execution fails
        execution_result = {"success": False, "stdout": "", "stderr": "tests failed"}

        # The verify block exists but has wrong assertions
        verify_block = _extract_verify_block(agent_response)
        assert verify_block is not None

        agent_verify_result = {"success": False, "error": "verify block has wrong assertions"}

        # Certify the episode
        hybrid_cert = certify_episode(
            trace_embeddings=stable_embeddings,
            execution_result=execution_result,
            spectral_certificate=spectral_cert,
            semantic_oracle_passed=True,
            semantic_compliance_score=1.0,
            agent_verify_result=agent_verify_result,
        )

        # **THE CRITICAL ASSERTION**: Certificate must be FAIL_EXECUTION despite stability
        verdict = hybrid_cert["certified_verdict"]
        assert verdict == CertifiedVerdict.FAIL_EXECUTION.value, (
            f"Expected FAIL_EXECUTION for stable-but-wrong agent, got {verdict}. "
            f"Spectral bound was {theoretical_bound:.4f} (low = stable). "
            f"This demonstrates Phase 3 gating logic working correctly."
        )

        # Verify the reasoning explains the failure
        reasoning = hybrid_cert.get("reasoning", "")
        assert "failed" in reasoning.lower() or "incompetent" in reasoning.lower()

    def test_stable_and_correct_gets_pass(self):
        """Verify that stable AND correct agents get PASS.

        Sanity check: make sure the gating logic doesn't reject all agents.
        """
        stable_embeddings = [
            [0.1, 0.2, 0.3],
            [0.11, 0.21, 0.31],
            [0.12, 0.22, 0.32],
        ]

        spectral_cert = compute_certificate(stable_embeddings)
        theoretical_bound = spectral_cert.get("theoretical_bound", float("inf"))
        assert theoretical_bound < 1.0

        execution_result = {"success": True, "stdout": "all tests passed"}
        agent_verify_result = {"success": True}

        hybrid_cert = certify_episode(
            trace_embeddings=stable_embeddings,
            execution_result=execution_result,
            spectral_certificate=spectral_cert,
            semantic_oracle_passed=True,
            semantic_compliance_score=1.0,
            agent_verify_result=agent_verify_result,
        )

        verdict = hybrid_cert["certified_verdict"]
        assert verdict == CertifiedVerdict.PASS.value, (
            f"Expected PASS for stable and correct agent, got {verdict}"
        )

    def test_unstable_rejected_even_if_lucky_on_tests(self):
        """Verify that drift is detected even if tests happen to pass.

        A drifting agent might get lucky and pass tests by accident,
        but Phase 3 still rejects it (FAIL_DRIFT).
        """
        # Create embeddings with high variance (drift)
        drifting_embeddings = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],  # Big jump
            [2.0, 0.0, 2.0],  # Random drift
            [0.5, 1.5, 0.5],  # Unstable
        ]

        spectral_cert = compute_certificate(drifting_embeddings)
        theoretical_bound = spectral_cert.get("theoretical_bound", float("inf"))

        # Manually set to high bound if not already (ensure drift)
        if theoretical_bound < 0.5:
            spectral_cert["theoretical_bound"] = 0.6

        execution_result = {"success": True, "stdout": "lucky pass"}
        agent_verify_result = None

        hybrid_cert = certify_episode(
            trace_embeddings=drifting_embeddings,
            execution_result=execution_result,
            spectral_certificate=spectral_cert,
            semantic_oracle_passed=True,
            semantic_compliance_score=1.0,
            agent_verify_result=agent_verify_result,
            bound_threshold=0.5,
        )

        verdict = hybrid_cert["certified_verdict"]
        assert verdict == CertifiedVerdict.FAIL_DRIFT.value, (
            f"Expected FAIL_DRIFT for drifting agent despite passing tests, got {verdict}. "
            f"Bound was {spectral_cert['theoretical_bound']:.4f}"
        )

    def test_semantic_failure_detected(self):
        """Verify that semantic oracle failures are detected."""
        stable_embeddings = [
            [0.1, 0.2, 0.3],
            [0.11, 0.21, 0.31],
        ]

        spectral_cert = compute_certificate(stable_embeddings)
        execution_result = {"success": True}

        hybrid_cert = certify_episode(
            trace_embeddings=stable_embeddings,
            execution_result=execution_result,
            spectral_certificate=spectral_cert,
            semantic_oracle_passed=False,  # Oracle flagged unsafe/incoherent steps
            semantic_compliance_score=0.3,
            agent_verify_result=None,
        )

        verdict = hybrid_cert["certified_verdict"]
        assert verdict == CertifiedVerdict.FAIL_SEMANTIC.value, (
            f"Expected FAIL_SEMANTIC when oracle fails, got {verdict}"
        )

    def test_robustness_failure_detected(self):
        """Verify that embedding instability (high lipschitz margin) is detected."""
        stable_embeddings = [
            [0.1, 0.2, 0.3],
            [0.11, 0.21, 0.31],
        ]

        spectral_cert = compute_certificate(stable_embeddings)
        spectral_cert["lipschitz_margin"] = 1.5  # High instability

        execution_result = {"success": True}

        hybrid_cert = certify_episode(
            trace_embeddings=stable_embeddings,
            execution_result=execution_result,
            spectral_certificate=spectral_cert,
            semantic_oracle_passed=True,
            semantic_compliance_score=1.0,
            agent_verify_result=None,
            robustness_threshold=1.0,
        )

        verdict = hybrid_cert["certified_verdict"]
        assert verdict == CertifiedVerdict.FAIL_ROBUSTNESS.value, (
            f"Expected FAIL_ROBUSTNESS when lipschitz margin high, got {verdict}"
        )


class TestExecutionWitness:
    """Test the ExecutionWitness collection and reporting."""

    def test_execution_witness_captured(self):
        """Verify that execution evidence is properly captured."""
        stable_embeddings = [[0.1, 0.2], [0.11, 0.21]]

        spectral_cert = compute_certificate(stable_embeddings)
        execution_result = {
            "success": False,
            "stdout": "test output",
            "stderr": "error message",
        }

        hybrid_cert = certify_episode(
            trace_embeddings=stable_embeddings,
            execution_result=execution_result,
            spectral_certificate=spectral_cert,
            semantic_oracle_passed=True,
        )

        # Check execution witness is populated
        witness = hybrid_cert.get("execution_witness", {})
        assert witness.get("ground_truth_passed") is False
        assert witness.get("semantic_oracle_passed") is True

    def test_agent_verify_result_in_witness(self):
        """Verify that agent-generated verify() block results are captured."""
        stable_embeddings = [[0.1, 0.2], [0.11, 0.21]]

        spectral_cert = compute_certificate(stable_embeddings)
        execution_result = {"success": True}

        agent_verify_result = {
            "success": False,
            "output": "assertion failed",
            "verify_code": "def verify():\n    assert False",
        }

        hybrid_cert = certify_episode(
            trace_embeddings=stable_embeddings,
            execution_result=execution_result,
            spectral_certificate=spectral_cert,
            semantic_oracle_passed=True,
            agent_verify_result=agent_verify_result,
        )

        witness = hybrid_cert.get("execution_witness", {})
        assert witness.get("agent_generated_passed") is False
        assert witness.get("agent_verify_block") == agent_verify_result["verify_code"]


class TestThresholdSensitivity:
    """Test how thresholds affect certification verdicts."""

    def test_bound_threshold_controls_drift_detection(self):
        """Verify that bound_threshold controls when FAIL_DRIFT occurs."""
        drifting_embeddings = [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
        ]

        spectral_cert = compute_certificate(drifting_embeddings)
        spectral_cert["theoretical_bound"] = 0.55

        execution_result = {"success": True}

        # With threshold 0.5, should fail
        hybrid_cert_strict = certify_episode(
            trace_embeddings=drifting_embeddings,
            execution_result=execution_result,
            spectral_certificate=spectral_cert,
            semantic_oracle_passed=True,
            bound_threshold=0.5,
        )
        assert hybrid_cert_strict["certified_verdict"] == CertifiedVerdict.FAIL_DRIFT.value

        # With threshold 0.6, should pass
        hybrid_cert_loose = certify_episode(
            trace_embeddings=drifting_embeddings,
            execution_result=execution_result,
            spectral_certificate=spectral_cert,
            semantic_oracle_passed=True,
            bound_threshold=0.6,
        )
        assert hybrid_cert_loose["certified_verdict"] == CertifiedVerdict.PASS.value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
