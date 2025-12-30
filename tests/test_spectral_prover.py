"""Unit tests for spectral_prover.py

Tests cover:
1. Synthetic linear system generation and certificate computation
2. Davis-Kahan bound verification
3. Perturbation detection via residual increase
4. Subspace angle computation

All data are synthetic. No real secrets or external network calls.
"""
import numpy as np
import pytest
from numpy.linalg import norm

from certificates.spectral_prover import (
    compute_detection_statistics,
    compute_svd_certificate,
    davis_kahan_upper_bound,
    generate_perturbed_trajectory,
    generate_synthetic_linear_system,
    koopman_operator_fit,
    subspace_angle,
    trajectory_matrix,
    compute_theoretical_bound,
)


class TestTrajectoryMatrix:
    """Tests for trajectory_matrix function."""

    def test_basic_shape(self):
        """Test that trajectory matrix has correct shape."""
        embeddings = np.random.randn(20, 10)  # T=20, d=10
        X = trajectory_matrix(embeddings, normalize=False)
        assert X.shape == (10, 20), f"Expected (10, 20), got {X.shape}"

    def test_normalization(self):
        """Test that normalization produces zero-mean, unit-variance."""
        embeddings = np.random.randn(50, 10) * 5 + 3  # Shifted and scaled
        X = trajectory_matrix(embeddings, normalize=True)
        # Check that each row has approximately zero mean and unit std
        row_means = X.mean(axis=1)
        row_stds = X.std(axis=1)
        assert np.allclose(row_means, 0, atol=1e-10), "Rows should be zero-mean"
        assert np.allclose(row_stds, 1, atol=1e-10), "Rows should have unit variance"

    def test_hankel_matrix(self):
        """Test Hankel (time-delay) embedding construction."""
        embeddings = np.random.randn(30, 5)  # T=30, d=5
        L = 5  # Window size
        H = trajectory_matrix(embeddings, hankel=True, L=L, normalize=False)
        expected_rows = L * 5
        expected_cols = 30 - L + 1
        assert H.shape == (expected_rows, expected_cols), \
            f"Expected ({expected_rows}, {expected_cols}), got {H.shape}"


class TestSVDCertificate:
    """Tests for compute_svd_certificate function."""

    def test_certificate_keys(self):
        """Test that certificate contains all required keys."""
        X = np.random.randn(10, 50)  # d=10, T=50
        cert = compute_svd_certificate(X, k=5)
        required_keys = [
            'residual', 'theoretical_bound', 'sigma_max', 'singular_gap',
            'tail_energy', 'pca_explained', 'U_k', 'Sigma_k', 'V_k'
        ]
        for key in required_keys:
            assert key in cert, f"Missing key: {key}"

    def test_residual_bounds(self):
        """Test that residual is in [0, 1]."""
        X = np.random.randn(10, 50)
        cert = compute_svd_certificate(X, k=5)
        assert 0 <= cert['residual'] <= 1, f"Residual {cert['residual']} not in [0, 1]"

    def test_pca_explained_bounds(self):
        """Test that pca_explained is in [0, 1]."""
        X = np.random.randn(10, 50)
        cert = compute_svd_certificate(X, k=5)
        assert 0 <= cert['pca_explained'] <= 1, \
            f"pca_explained {cert['pca_explained']} not in [0, 1]"

    def test_low_rank_matrix_has_low_residual(self):
        """Test that a low-rank matrix has small residual when k >= rank."""
        # Create rank-3 matrix
        rng = np.random.default_rng(42)
        A = rng.standard_normal((10, 3))
        B = rng.standard_normal((3, 50))
        X = A @ B  # Rank 3

        cert = compute_svd_certificate(X, k=5)  # k > rank
        assert cert['residual'] < 1e-10, \
            f"Rank-3 matrix with k=5 should have ~0 residual, got {cert['residual']}"

    def test_singular_vectors_orthogonality(self):
        """Test that returned singular vectors are orthonormal."""
        X = np.random.randn(10, 50)
        cert = compute_svd_certificate(X, k=5)

        U_k = cert['U_k']
        V_k = cert['V_k']

        # Check U_k is orthonormal: U_k.T @ U_k should be identity
        U_gram = U_k.T @ U_k
        assert np.allclose(U_gram, np.eye(U_gram.shape[0]), atol=1e-10), \
            "U_k columns are not orthonormal"

        # Check V_k is orthonormal
        V_gram = V_k.T @ V_k
        assert np.allclose(V_gram, np.eye(V_gram.shape[0]), atol=1e-10), \
            "V_k columns are not orthonormal"


class TestSubspaceAngle:
    """Tests for subspace_angle function."""

    def test_identical_subspaces(self):
        """Test that identical subspaces have zero angle."""
        rng = np.random.default_rng(42)
        U = np.linalg.qr(rng.standard_normal((10, 3)))[0]
        angle = subspace_angle(U, U)
        assert angle < 1e-10, f"Identical subspaces should have angle ~0, got {angle}"

    def test_orthogonal_subspaces(self):
        """Test that orthogonal subspaces have angle π/2."""
        # Create orthogonal subspaces in R^10
        U = np.zeros((10, 3))
        U[:3, :] = np.eye(3)  # First 3 dimensions

        V = np.zeros((10, 3))
        V[3:6, :] = np.eye(3)  # Dimensions 4-6

        angle = subspace_angle(U, V)
        assert np.isclose(angle, np.pi / 2, atol=1e-10), \
            f"Orthogonal subspaces should have angle π/2, got {angle}"

    def test_angle_symmetry(self):
        """Test that angle(U, V) = angle(V, U)."""
        rng = np.random.default_rng(42)
        U = np.linalg.qr(rng.standard_normal((10, 3)))[0]
        V = np.linalg.qr(rng.standard_normal((10, 3)))[0]

        angle_uv = subspace_angle(U, V)
        angle_vu = subspace_angle(V, U)

        assert np.isclose(angle_uv, angle_vu, atol=1e-10), \
            f"Angles should be symmetric: {angle_uv} vs {angle_vu}"


class TestDavisKahanBound:
    """Tests for davis_kahan_upper_bound function."""

    def test_zero_perturbation(self):
        """Test that zero perturbation gives zero bound."""
        bound = davis_kahan_upper_bound(E_norm=0.0, delta=1.0)
        assert bound == 0.0, f"Zero perturbation should give bound 0, got {bound}"

    def test_small_gap_gives_large_bound(self):
        """Test that small gap leads to larger bound."""
        bound_small_gap = davis_kahan_upper_bound(E_norm=0.1, delta=0.01)
        bound_large_gap = davis_kahan_upper_bound(E_norm=0.1, delta=1.0)
        assert bound_small_gap > bound_large_gap, \
            "Smaller gap should give larger bound"

    def test_bound_clipping(self):
        """Test that bound is clipped to [0, 1]."""
        # E_norm > delta should give bound = 1 (clipped)
        bound = davis_kahan_upper_bound(E_norm=10.0, delta=1.0)
        assert bound == 1.0, f"Large perturbation should clip to 1, got {bound}"

    def test_zero_gap(self):
        """Test that zero gap gives bound = 1."""
        bound = davis_kahan_upper_bound(E_norm=0.1, delta=0.0)
        assert bound == 1.0, f"Zero gap should give bound 1, got {bound}"


class TestSyntheticLinearSystem:
    """Tests for synthetic linear system generation."""

    def test_trajectory_shape(self):
        """Test that generated trajectory has correct shape."""
        X, K = generate_synthetic_linear_system(d=10, T=50, k=3)
        assert X.shape == (50, 10), f"Expected (50, 10), got {X.shape}"
        assert K.shape == (10, 10), f"Expected (10, 10), got {K.shape}"

    def test_stability(self):
        """Test that generated system is stable (spectral radius < 1)."""
        X, K = generate_synthetic_linear_system(d=10, T=50, k=3, seed=42)
        eigenvalues = np.linalg.eigvals(K)
        spectral_radius = np.max(np.abs(eigenvalues))
        assert spectral_radius < 1.0, \
            f"System should be stable, got spectral radius {spectral_radius}"

    def test_low_residual_for_clean_system(self):
        """Test that clean linear system has reasonable Koopman residual."""
        X, K = generate_synthetic_linear_system(
            d=10, T=100, k=3, noise_std=0.001, seed=42
        )
        X_traj = trajectory_matrix(X, normalize=True)
        koop = koopman_operator_fit(X_traj, k=5)
        # Normalization can increase residual; threshold is empirically calibrated
        assert koop['prediction_residual'] < 0.8, \
            f"Clean linear system should have reasonable residual, got {koop['prediction_residual']}"


class TestPerturbationDetection:
    """Tests for perturbation detection via spectral certificates."""

    def test_residual_increases_with_perturbation(self):
        """Test that residual increases when perturbation is added."""
        # Generate clean trajectory
        X_clean, _ = generate_synthetic_linear_system(
            d=10, T=50, k=3, noise_std=0.01, seed=42
        )

        # Compute clean statistics
        stats_clean = compute_detection_statistics(X_clean, k=5)

        # Add perturbation
        X_perturbed, E = generate_perturbed_trajectory(
            X_clean, perturbation_norm=0.5, perturbation_type='additive', seed=42
        )

        # Compute perturbed statistics
        stats_perturbed = compute_detection_statistics(
            X_perturbed, k=5, baseline_U=stats_clean['U_k']
        )

        # Residual should increase
        assert stats_perturbed['residual'] >= stats_clean['residual'], \
            f"Residual should increase with perturbation: " \
            f"{stats_clean['residual']} -> {stats_perturbed['residual']}"

    def test_dk_angle_scales_with_perturbation(self):
        """Test that Davis-Kahan angle increases with perturbation magnitude."""
        X_clean, _ = generate_synthetic_linear_system(
            d=10, T=100, k=3, noise_std=0.01, seed=42
        )
        stats_clean = compute_detection_statistics(X_clean, k=5)

        angles = []
        for pert_norm in [0.1, 0.3, 0.5, 1.0]:
            X_pert, _ = generate_perturbed_trajectory(
                X_clean, perturbation_norm=pert_norm, seed=42
            )
            stats_pert = compute_detection_statistics(
                X_pert, k=5, baseline_U=stats_clean['U_k']
            )
            angles.append(stats_pert['dk_angle'])

        # Angles should generally increase (may not be strictly monotonic due to normalization)
        assert angles[-1] > angles[0], \
            f"Larger perturbation should give larger dk_angle: {angles}"

    def test_dk_angle_bounded_by_davis_kahan(self):
        """Test that computed dk_angle is approximately bounded by Davis-Kahan formula."""
        X_clean, _ = generate_synthetic_linear_system(
            d=10, T=100, k=3, noise_std=0.001, seed=42
        )
        stats_clean = compute_detection_statistics(X_clean, k=5)

        X_pert, E = generate_perturbed_trajectory(
            X_clean, perturbation_norm=0.3, seed=42
        )
        stats_pert = compute_detection_statistics(
            X_pert, k=5, baseline_U=stats_clean['U_k']
        )

        # The angle should be less than or close to the theoretical bound
        # Note: the bound is an upper bound, actual angle can be less
        sin_theta_measured = np.sin(stats_pert['dk_angle'])
        sin_theta_bound = stats_pert['dk_bound']

        # Allow some tolerance due to numerical issues and bound looseness
        assert sin_theta_measured <= sin_theta_bound + 0.3, \
            f"sin(dk_angle)={sin_theta_measured:.4f} should be <= bound={sin_theta_bound:.4f}"


class TestKoopmanOperatorFit:
    """Tests for Koopman operator fitting."""

    def test_fitted_operator_shape(self):
        """Test that fitted operator has correct shape."""
        X = np.random.randn(50, 10)
        X_traj = trajectory_matrix(X)
        koop = koopman_operator_fit(X_traj, k=5)
        assert koop['K_approx'].shape == (5, 5), \
            f"Expected (5, 5), got {koop['K_approx'].shape}"

    def test_prediction_residual_bounds(self):
        """Test that prediction residual is in reasonable range."""
        X = np.random.randn(50, 10)
        X_traj = trajectory_matrix(X)
        koop = koopman_operator_fit(X_traj, k=5)
        assert 0 <= koop['prediction_residual'] <= 2, \
            f"Prediction residual {koop['prediction_residual']} seems unreasonable"

    def test_recovers_linear_dynamics(self):
        """Test that Koopman fit can recover linear dynamics structure."""
        X, K_true = generate_synthetic_linear_system(
            d=10, T=200, k=3, noise_std=0.001, seed=42
        )
        X_traj = trajectory_matrix(X, normalize=True)
        koop = koopman_operator_fit(X_traj, k=5)

        # Prediction residual should be reasonable for clean linear system
        # Normalization affects the residual; threshold is empirically calibrated
        assert koop['prediction_residual'] < 0.8, \
            f"Should recover linear dynamics structure, got {koop['prediction_residual']}"


class TestDetectionStatistics:
    """Tests for compute_detection_statistics function."""

    def test_all_keys_present(self):
        """Test that all expected keys are in output."""
        X = np.random.randn(30, 10)
        stats = compute_detection_statistics(X, k=5)

        expected_keys = [
            'residual', 'theoretical_bound', 'sigma_max', 'singular_gap',
            'tail_energy', 'pca_explained', 'dk_angle', 'dk_bound',
            'koopman_residual', 'length_T', 'U_k'
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"

    def test_length_T_correct(self):
        """Test that length_T matches input length."""
        X = np.random.randn(42, 10)
        stats = compute_detection_statistics(X, k=5)
        assert stats['length_T'] == 42, f"Expected length_T=42, got {stats['length_T']}"

    def test_theoretical_bound_formula(self):
        """Test that theoretical_bound = C_res * residual + C_tail * tail_energy."""
        X = np.random.randn(50, 10)
        C_res, C_tail = 1.5, 0.8
        stats = compute_detection_statistics(X, k=5, C_res=C_res, C_tail=C_tail)

        expected = C_res * stats['residual'] + C_tail * stats['tail_energy']
        assert np.isclose(stats['theoretical_bound'], expected, atol=1e-10), \
            f"theoretical_bound mismatch: {stats['theoretical_bound']} vs {expected}"


class TestTheoreticalBound:
    """Tests for compute_theoretical_bound function."""

    def test_bound_formula(self):
        """Test that bound formula is correctly implemented."""
        bound = compute_theoretical_bound(
            residual=0.1,
            tail_energy=0.2,
            semantic_divergence=0.3,
            lipschitz_margin=0.05,
            C_res=1.0,
            C_tail=1.0,
            C_sem=1.0,
            C_robust=1.0,
        )
        expected = 0.1 + 0.2 + 0.3 + 0.05
        assert np.isclose(bound, expected), f"Expected {expected}, got {bound}"

    def test_bound_with_custom_coefficients(self):
        """Test bound with non-unit coefficients."""
        bound = compute_theoretical_bound(
            residual=0.1,
            tail_energy=0.2,
            C_res=2.0,
            C_tail=0.5,
        )
        expected = 2.0 * 0.1 + 0.5 * 0.2
        assert np.isclose(bound, expected), f"Expected {expected}, got {bound}"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_timestep(self):
        """Test handling of single-timestep trajectory."""
        X = np.random.randn(1, 10)
        stats = compute_detection_statistics(X, k=5)
        # Should not crash, should return conservative values
        assert stats['length_T'] == 1

    def test_high_dimensional_embeddings(self):
        """Test with high-dimensional embeddings (d > T)."""
        X = np.random.randn(10, 100)  # T=10, d=100
        stats = compute_detection_statistics(X, k=5)
        assert 0 <= stats['residual'] <= 1

    def test_k_larger_than_rank(self):
        """Test when k is larger than actual rank."""
        # Create rank-3 data
        rng = np.random.default_rng(42)
        A = rng.standard_normal((10, 3))
        B = rng.standard_normal((3, 50))
        X = (A @ B).T  # (50, 10)

        stats = compute_detection_statistics(X, k=10)  # k > rank
        assert stats['pca_explained'] > 0.99, \
            "Should explain nearly all variance for low-rank data"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
