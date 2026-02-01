"""Tests for per-step diagnostics computation.

These tests verify:
1. Off-manifold ratio computation
2. Per-step residual computation
3. Subspace angle computation
4. Operator norm computation
"""
import numpy as np
import pytest

# Mark all tests as unit tests
pytestmark = pytest.mark.unit


class TestOffManifoldRatio:
    """Tests for per-step off-manifold ratio computation."""

    def test_off_manifold_basic(self):
        """Test basic off-manifold computation."""
        from certificates.make_certificate import compute_per_step_off_manifold

        # Create data with known structure
        T, D = 10, 5
        X_aug = np.random.randn(T, D + 1)

        # Create orthonormal basis
        U, S, Vt = np.linalg.svd(X_aug, full_matrices=False)
        V_r = Vt[:3, :].T  # (D+1, 3)

        off_ratios = compute_per_step_off_manifold(X_aug, V_r)

        # Check output shape
        assert len(off_ratios) == T

        # Check values in valid range
        for ratio in off_ratios:
            assert 0 <= ratio <= 1

    def test_off_manifold_on_subspace(self):
        """Test that points on subspace have zero off-manifold ratio."""
        from certificates.make_certificate import compute_per_step_off_manifold

        # Create data that lies exactly in a subspace
        T, r = 10, 3
        # Generate data in r-dimensional subspace
        Z = np.random.randn(T, r)
        V_r = np.eye(r + 1, r)  # Identity projection
        X_aug = Z @ V_r.T  # Data lies in V_r subspace

        off_ratios = compute_per_step_off_manifold(X_aug, V_r)

        # All ratios should be near zero
        for ratio in off_ratios:
            assert ratio < 1e-10, f"Expected near-zero, got {ratio}"

    def test_off_manifold_orthogonal(self):
        """Test that orthogonal data has high off-manifold ratio."""
        from certificates.make_certificate import compute_per_step_off_manifold

        # Create data orthogonal to subspace
        T = 10
        X_aug = np.zeros((T, 4))
        X_aug[:, 3] = np.random.randn(T)  # Only in 4th dimension

        V_r = np.eye(4, 3)  # Subspace spans first 3 dimensions

        off_ratios = compute_per_step_off_manifold(X_aug, V_r)

        # All ratios should be 1.0 (completely off-manifold)
        for ratio in off_ratios:
            assert ratio > 0.99, f"Expected ~1.0, got {ratio}"


class TestPerStepResiduals:
    """Tests for per-step residual computation."""

    def test_residuals_basic(self):
        """Test basic residual computation."""
        from certificates.make_certificate import compute_per_step_residuals

        T, r = 10, 3
        Z = np.random.randn(T, r)
        A = np.random.randn(r, r)

        r_norms = compute_per_step_residuals(Z, A)

        # Check output shape
        assert len(r_norms) == T

        # Last element should be zero (no prediction)
        assert r_norms[-1] == 0.0

        # All values should be non-negative
        for r_norm in r_norms:
            assert r_norm >= 0

    def test_residuals_perfect_prediction(self):
        """Test that perfect prediction gives zero residuals."""
        from certificates.make_certificate import compute_per_step_residuals

        T, r = 10, 3

        # Create data that follows perfect linear dynamics
        A = np.eye(r) * 0.9  # Simple decay
        Z = np.zeros((T, r))
        Z[0] = np.random.randn(r)
        for t in range(1, T):
            Z[t] = A @ Z[t - 1]

        r_norms = compute_per_step_residuals(Z, A)

        # All residuals should be near zero (except last)
        for r_norm in r_norms[:-1]:
            assert r_norm < 1e-10, f"Expected near-zero, got {r_norm}"


class TestSubspaceAngles:
    """Tests for subspace angle (sin theta) computation."""

    def test_sin_theta_identical_subspaces(self):
        """Test that identical subspaces have zero angle."""
        from certificates.make_certificate import compute_sin_theta

        U = np.eye(10, 5)
        sin_max, sin_fro = compute_sin_theta(U, U)

        assert sin_max < 1e-10
        assert sin_fro < 1e-10

    def test_sin_theta_orthogonal_subspaces(self):
        """Test that orthogonal subspaces have maximal angle."""
        from certificates.make_certificate import compute_sin_theta

        # Two orthogonal subspaces
        U1 = np.zeros((10, 3))
        U1[:3, :] = np.eye(3)  # First 3 dimensions
        U2 = np.zeros((10, 3))
        U2[3:6, :] = np.eye(3)  # Dimensions 4-6

        sin_max, sin_fro = compute_sin_theta(U1, U2)

        # Orthogonal subspaces should have sin(theta) = 1
        assert sin_max > 0.99

    def test_sin_theta_small_perturbation(self):
        """Test that small perturbation gives small angle."""
        from certificates.make_certificate import compute_sin_theta

        U1 = np.eye(10, 5)

        # Small perturbation
        eps = 0.01
        perturbation = eps * np.random.randn(10, 5)
        U2_raw = U1 + perturbation
        # Re-orthogonalize
        U2, _ = np.linalg.qr(U2_raw)

        sin_max, sin_fro = compute_sin_theta(U1, U2)

        # Angle should be small (proportional to perturbation)
        assert sin_max < 0.1, f"Expected small angle, got sin_max={sin_max}"


class TestOperatorNorm:
    """Tests for operator difference norm computation."""

    def test_E_norm_identical(self):
        """Test that identical operators have zero norm."""
        from certificates.make_certificate import compute_E_norm

        A = np.random.randn(5, 5)
        E_norm = compute_E_norm(A, A)

        assert E_norm < 1e-10

    def test_E_norm_frobenius(self):
        """Test Frobenius norm computation."""
        from certificates.make_certificate import compute_E_norm

        A1 = np.eye(3)
        A2 = np.zeros((3, 3))

        # ||I - 0||_F = sqrt(3)
        E_norm = compute_E_norm(A1, A2, norm_type="fro")
        expected = np.sqrt(3)

        np.testing.assert_almost_equal(E_norm, expected, decimal=10)

    def test_E_norm_spectral(self):
        """Test spectral norm computation."""
        from certificates.make_certificate import compute_E_norm

        A1 = np.eye(3)
        A2 = np.zeros((3, 3))

        # ||I - 0||_2 = 1 (largest singular value)
        E_norm = compute_E_norm(A1, A2, norm_type="2")

        np.testing.assert_almost_equal(E_norm, 1.0, decimal=10)


class TestIntegrationWithCertificate:
    """Integration tests with full certificate computation."""

    def test_per_step_diagnostics_shape_consistency(self):
        """Test that per-step diagnostics have consistent shapes."""
        from certificates.make_certificate import (
            compute_per_step_off_manifold,
            compute_per_step_residuals,
            _fit_temporal_operator_ridge,
        )

        T, D = 25, 64
        X = np.random.randn(T, D)
        X_aug = np.concatenate([X, np.ones((T, 1))], axis=1)

        # SVD
        U, S, Vt = np.linalg.svd(X_aug, full_matrices=False)
        r_eff = 5
        V_r = Vt[:r_eff, :].T
        Z = X_aug @ V_r

        # Fit operator
        X0 = Z[:-1].T
        X1 = Z[1:].T
        A = _fit_temporal_operator_ridge(X0, X1)

        # Compute diagnostics
        off_ratios = compute_per_step_off_manifold(X_aug, V_r)
        r_norms = compute_per_step_residuals(Z, A)

        assert len(off_ratios) == T
        assert len(r_norms) == T
        assert all(0 <= r <= 1 for r in off_ratios)
        assert all(r >= 0 for r in r_norms)
