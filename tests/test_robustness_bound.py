"""Verification tests for Lipschitz robustness margin in theoretical bound.

This module tests that the lipschitz_margin term correctly penalizes
embedding instability in the spectral certificate computation.
"""

import numpy as np
import pytest

from certificates.make_certificate import compute_certificate
from certificates import uelat_bridge


def create_dummy_trace_embeddings(n_steps: int = 10, dim: int = 32) -> np.ndarray:
    """Create a simple dummy trace of embeddings for testing.

    Parameters
    ----------
    n_steps : int
        Number of timesteps in the trace.
    dim : int
        Dimension of each embedding.

    Returns
    -------
    np.ndarray
        Array of shape (n_steps, dim) with normalized embeddings.
    """
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_steps, dim))
    # Normalize embeddings
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms
    return X


class TestLipschitzMarginInBound:
    """Test that lipschitz_margin correctly affects the theoretical bound."""

    def test_lipschitz_margin_increases_bound(self):
        """Assert: bound_with_margin > bound_baseline."""
        embeddings = create_dummy_trace_embeddings(n_steps=15, dim=32)

        # Compute baseline certificate without margin
        cert_baseline = compute_certificate(embeddings, r=5, lipschitz_margin=0.0)
        baseline_bound = cert_baseline["theoretical_bound"]

        # Compute certificate with nonzero margin
        margin_value = 0.5
        cert_with_margin = compute_certificate(embeddings, r=5, lipschitz_margin=margin_value)
        bound_with_margin = cert_with_margin["theoretical_bound"]

        # Check that adding margin increases the bound
        assert bound_with_margin > baseline_bound, (
            f"Expected bound_with_margin ({bound_with_margin}) > baseline ({baseline_bound})"
        )

    def test_lipschitz_margin_penalty_is_correct(self):
        """Assert: increase is exactly C_robust * lipschitz_margin."""
        embeddings = create_dummy_trace_embeddings(n_steps=15, dim=32)
        margin_value = 0.5

        # Compute baselines
        cert_baseline = compute_certificate(embeddings, r=5, lipschitz_margin=0.0)
        baseline_bound = cert_baseline["theoretical_bound"]
        c_robust_baseline = cert_baseline.get("C_robust", 1.0)

        # Compute with margin
        cert_with_margin = compute_certificate(embeddings, r=5, lipschitz_margin=margin_value)
        bound_with_margin = cert_with_margin["theoretical_bound"]
        c_robust_margin = cert_with_margin.get("C_robust", 1.0)

        # Verify C_robust is consistent
        assert c_robust_baseline == c_robust_margin, (
            f"C_robust should be consistent: {c_robust_baseline} != {c_robust_margin}"
        )

        # Verify the penalty is exact: increase = C_robust * margin
        expected_increase = c_robust_baseline * margin_value
        actual_increase = bound_with_margin - baseline_bound

        # Use relative tolerance to account for floating-point arithmetic
        rel_tol = 1e-10
        abs_tol = 1e-12
        assert np.isclose(actual_increase, expected_increase, rtol=rel_tol, atol=abs_tol), (
            f"Penalty mismatch: expected increase {expected_increase}, got {actual_increase}"
        )

    def test_lipschitz_margin_in_certificate_dict(self):
        """Assert: lipschitz_margin is included in returned certificate."""
        embeddings = create_dummy_trace_embeddings(n_steps=10, dim=32)
        margin_value = 0.3

        cert = compute_certificate(embeddings, r=5, lipschitz_margin=margin_value)

        # Check that lipschitz_margin is in the certificate
        assert "lipschitz_margin" in cert, "lipschitz_margin not found in certificate"
        assert cert["lipschitz_margin"] == margin_value, (
            f"Expected lipschitz_margin={margin_value}, got {cert['lipschitz_margin']}"
        )

    def test_c_robust_constant_in_certificate(self):
        """Assert: C_robust constant is included and valid."""
        embeddings = create_dummy_trace_embeddings(n_steps=10, dim=32)

        cert = compute_certificate(embeddings, r=5, lipschitz_margin=0.2)

        # Check that C_robust is in the certificate
        assert "C_robust" in cert, "C_robust constant not found in certificate"
        c_robust = cert["C_robust"]

        # Verify it's finite and within bounds
        assert np.isfinite(c_robust), f"C_robust not finite: {c_robust}"
        assert c_robust > 0, f"C_robust must be positive: {c_robust}"
        assert c_robust <= 2.0, f"C_robust exceeds proven bound: {c_robust}"

    def test_zero_margin_no_penalty(self):
        """Assert: zero margin adds no penalty."""
        embeddings = create_dummy_trace_embeddings(n_steps=10, dim=32)

        cert_zero = compute_certificate(embeddings, r=5, lipschitz_margin=0.0)
        cert_near_zero = compute_certificate(embeddings, r=5, lipschitz_margin=1e-15)

        # Both should have virtually identical bounds
        assert np.isclose(cert_zero["theoretical_bound"], cert_near_zero["theoretical_bound"]), (
            "Zero margin and near-zero margin should give same bound"
        )

    def test_negative_margin_not_applied(self):
        """Assert: negative margin is clamped to 0.0 or handled gracefully."""
        embeddings = create_dummy_trace_embeddings(n_steps=10, dim=32)

        # The specification says lipschitz_margin must be non-negative
        # Some implementations may accept negative and clamp, or the caller ensures non-negative
        # This test documents the behavior
        try:
            cert = compute_certificate(embeddings, r=5, lipschitz_margin=-0.1)
            # If we get here, the function accepted it; verify it's not making it worse
            cert_baseline = compute_certificate(embeddings, r=5, lipschitz_margin=0.0)
            assert cert["theoretical_bound"] >= cert_baseline["theoretical_bound"], (
                "Negative margin should not decrease bound"
            )
        except ValueError:
            # It's also acceptable to reject negative values
            pass

    def test_large_margin_produces_large_penalty(self):
        """Assert: large margin produces proportionally large penalty."""
        embeddings = create_dummy_trace_embeddings(n_steps=10, dim=32)

        cert_baseline = compute_certificate(embeddings, r=5, lipschitz_margin=0.0)
        cert_small = compute_certificate(embeddings, r=5, lipschitz_margin=0.1)
        cert_large = compute_certificate(embeddings, r=5, lipschitz_margin=1.0)

        increase_small = cert_small["theoretical_bound"] - cert_baseline["theoretical_bound"]
        increase_large = cert_large["theoretical_bound"] - cert_baseline["theoretical_bound"]

        # Larger margin should produce larger increase (proportional)
        assert increase_large > increase_small, (
            f"Large margin ({1.0}) should produce larger increase "
            f"than small margin ({0.1})"
        )

        # Verify proportionality: increase should scale linearly with margin
        c_robust = cert_baseline.get("C_robust", 1.0)
        expected_ratio = 1.0 / 0.1
        actual_ratio = increase_large / (increase_small + 1e-12)
        rel_tol = 1e-8
        assert np.isclose(actual_ratio, expected_ratio, rtol=rel_tol), (
            f"Penalty should scale linearly: expected ratio {expected_ratio}, "
            f"got {actual_ratio}"
        )


class TestBackwardCompatibility:
    """Test that existing code without lipschitz_margin still works."""

    def test_default_lipschitz_margin_is_zero(self):
        """Assert: calling without lipschitz_margin defaults to 0.0."""
        embeddings = create_dummy_trace_embeddings(n_steps=10, dim=32)

        # Call without specifying lipschitz_margin
        cert_default = compute_certificate(embeddings, r=5)

        # Should have lipschitz_margin=0.0
        assert cert_default["lipschitz_margin"] == 0.0, (
            f"Default lipschitz_margin should be 0.0, got {cert_default['lipschitz_margin']}"
        )

    def test_backward_compatible_signature(self):
        """Assert: old code calling compute_certificate still works."""
        embeddings = create_dummy_trace_embeddings(n_steps=10, dim=32)

        # Old-style call with only embeddings and r
        cert = compute_certificate(embeddings, r=10)
        assert "theoretical_bound" in cert
        assert "residual" in cert
        assert "tail_energy" in cert
        assert "semantic_divergence" in cert
        assert "lipschitz_margin" in cert
        assert "C_robust" in cert

    def test_all_certificate_keys_present(self):
        """Assert: certificate contains all expected keys."""
        embeddings = create_dummy_trace_embeddings(n_steps=10, dim=32)
        cert = compute_certificate(embeddings, r=5, lipschitz_margin=0.3)

        expected_keys = {
            "theoretical_bound",
            "residual",
            "tail_energy",
            "semantic_divergence",
            "lipschitz_margin",
            "pca_explained",
            "sigma_max",
            "sigma_second",
            "singular_gap",
            "temporal_sigma_max",
            "temporal_sigma_second",
            "temporal_singular_gap",
            "C_res",
            "C_tail",
            "C_sem",
            "C_robust",
        }

        for key in expected_keys:
            assert key in cert, f"Expected key '{key}' not found in certificate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
