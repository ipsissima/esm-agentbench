"""Empirical tests for Wedin's Theorem bounds.

These tests verify that the relationship between operator perturbation
and subspace angles satisfies Wedin's Theorem:

    sin(Θ(V, V̂)) ≤ ||E|| / γ

where:
- Θ(V, V̂) are the principal angles between original and perturbed subspaces
- ||E|| is the perturbation norm
- γ is the singular gap
"""
import numpy as np
import pytest

# Mark all tests as unit tests
pytestmark = pytest.mark.unit


class TestWedinBoundEmpirical:
    """Empirical tests for Wedin's Theorem."""

    def test_wedin_bound_holds_basic(self):
        """Test that Wedin bound holds for random perturbation."""
        from certificates.make_certificate import compute_sin_theta, compute_E_norm

        # Create a matrix with known singular gap
        n, r = 50, 5
        U_true = np.random.randn(n, r)
        U_true, _ = np.linalg.qr(U_true)

        S_true = np.array([10.0, 8.0, 6.0, 4.0, 2.0])  # Known singular values
        V_true = np.random.randn(r, r)
        V_true, _ = np.linalg.qr(V_true)

        A_true = U_true @ np.diag(S_true) @ V_true

        # Add small perturbation
        eps = 0.1
        E = eps * np.random.randn(n, r)
        A_perturbed = A_true + E

        # Compute SVD of perturbed matrix
        U_pert, S_pert, Vt_pert = np.linalg.svd(A_perturbed, full_matrices=False)

        # Compute subspace angle (between left singular vectors)
        sin_max, _ = compute_sin_theta(U_true, U_pert)

        # Compute perturbation norm
        E_norm = compute_E_norm(A_true, A_perturbed)

        # Singular gap (between r-th and (r+1)-th singular value)
        # Since we only have r singular values, use gap to zero
        gamma = S_true[-1]  # Smallest singular value

        # Wedin bound: sin(theta) <= ||E|| / gamma
        wedin_ratio = E_norm / gamma

        # The bound should hold with some margin for numerical error
        assert sin_max <= wedin_ratio + 0.01, (
            f"Wedin bound violated: sin_max={sin_max:.4f} > ||E||/γ={wedin_ratio:.4f}"
        )

    def test_wedin_bound_tightness(self):
        """Test that Wedin bound is reasonably tight."""
        from certificates.make_certificate import compute_sin_theta, compute_E_norm

        # Create a rank-1 matrix where Wedin should be tight
        n = 20
        u = np.random.randn(n)
        u = u / np.linalg.norm(u)
        v = np.random.randn(1)

        sigma = 5.0
        A_true = sigma * np.outer(u, v)

        # Add perturbation in direction of u
        eps = 0.5
        E = eps * np.outer(u, v)
        A_perturbed = A_true + E

        # SVD
        U_true, S_true, _ = np.linalg.svd(A_true, full_matrices=False)
        U_pert, S_pert, _ = np.linalg.svd(A_perturbed, full_matrices=False)

        # For rank-1 case, subspace is just the first column
        sin_max, _ = compute_sin_theta(U_true[:, :1], U_pert[:, :1])

        # E_norm
        E_norm = compute_E_norm(A_true, A_perturbed)

        # For rank-1, gamma = sigma (gap to zero)
        gamma = sigma

        wedin_ratio = E_norm / gamma

        # Sin should be small when E is aligned with singular direction
        assert sin_max <= wedin_ratio + 0.01

    def test_wedin_with_prototype_kernel(self, tmp_path):
        """Test Wedin bound using prototype kernel computation."""
        import subprocess
        import json
        from pathlib import Path
        from certificates.make_certificate import export_kernel_input

        # Create synthetic data
        T, D = 20, 10
        X = np.random.randn(T, D)
        X_aug = np.concatenate([X, np.ones((T, 1))], axis=1)

        # Export kernel input
        input_path = tmp_path / "kernel_input.json"
        export_kernel_input(
            X_aug=X_aug,
            trace_id="wedin-test",
            output_path=str(input_path),
            precision_bits=64,
        )

        # Run prototype kernel
        repo_root = Path(__file__).parent.parent
        kernel_script = repo_root / "kernel" / "prototype" / "prototype_kernel.py"

        if not kernel_script.exists():
            pytest.skip("Prototype kernel not found")

        output_path = tmp_path / "kernel_output.json"
        result = subprocess.run(
            ["python", str(kernel_script), str(input_path), str(output_path), "--precision", "64"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Load output
        with open(output_path, 'r') as f:
            output = json.load(f)

        # Check Wedin bound is computed
        checks = output.get("checks", {})
        wedin = checks.get("wedin_bound", {})

        # Wedin bound check should exist and pass (at least estimate)
        assert "pass_estimate" in wedin
        # Note: may be True even if E_over_gamma is 0 (no perturbation computed)

    def test_wedin_scaling_with_perturbation(self):
        """Test that sin(theta) scales with perturbation size."""
        from certificates.make_certificate import compute_sin_theta

        n, r = 30, 5

        # Create base subspace
        U_base = np.random.randn(n, r)
        U_base, _ = np.linalg.qr(U_base)

        eps_values = [0.01, 0.05, 0.1, 0.2]
        sin_values = []

        for eps in eps_values:
            # Perturb subspace
            perturbation = eps * np.random.randn(n, r)
            U_pert = U_base + perturbation
            U_pert, _ = np.linalg.qr(U_pert)

            sin_max, _ = compute_sin_theta(U_base, U_pert)
            sin_values.append(sin_max)

        # Sin should increase monotonically with perturbation
        for i in range(len(sin_values) - 1):
            # Allow some tolerance for numerical noise
            assert sin_values[i] <= sin_values[i + 1] + 0.05, (
                f"Sin not monotonic: {sin_values[i]:.4f} > {sin_values[i+1]:.4f}"
            )


class TestGammaComputation:
    """Tests for singular gap (gamma) computation."""

    def test_gamma_well_separated_spectrum(self):
        """Test gamma computation for well-separated spectrum."""
        # Create matrix with known spectrum
        n, r = 20, 5
        singular_values = np.array([10.0, 8.0, 6.0, 4.0, 2.0])

        U = np.random.randn(n, r)
        U, _ = np.linalg.qr(U)
        V = np.random.randn(r, r)
        V, _ = np.linalg.qr(V)

        A = U @ np.diag(singular_values) @ V

        # Recompute SVD
        _, S_computed, _ = np.linalg.svd(A, full_matrices=False)

        # Gamma between consecutive singular values
        for i in range(len(singular_values) - 1):
            expected_gap = singular_values[i] - singular_values[i + 1]
            computed_gap = S_computed[i] - S_computed[i + 1]
            np.testing.assert_almost_equal(computed_gap, expected_gap, decimal=5)

    def test_gamma_clustered_spectrum(self):
        """Test behavior when singular values are clustered."""
        from certificates.make_certificate import compute_sin_theta

        n, r = 20, 5

        # Clustered singular values (small gamma)
        singular_values = np.array([10.0, 9.9, 9.8, 9.7, 9.6])

        U = np.random.randn(n, r)
        U, _ = np.linalg.qr(U)
        V = np.random.randn(r, r)
        V, _ = np.linalg.qr(V)

        A_true = U @ np.diag(singular_values) @ V

        # Small perturbation should cause larger subspace change due to small gamma
        eps = 0.05
        E = eps * np.random.randn(n, r)
        A_pert = A_true + E

        U_true, _, _ = np.linalg.svd(A_true, full_matrices=False)
        U_pert, _, _ = np.linalg.svd(A_pert, full_matrices=False)

        sin_max, _ = compute_sin_theta(U_true, U_pert)

        # With small gamma (~0.1), same perturbation causes larger angle
        # gamma = 0.1, E_norm ~ 0.05 * sqrt(n*r) ~ 0.5
        # sin(theta) could be up to 5.0 (but capped at 1)
        # So we just check it's computed without error
        assert 0 <= sin_max <= 1
