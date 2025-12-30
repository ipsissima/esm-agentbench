#!/usr/bin/env python3
"""Spectral math regression tests using synthetic matrices.

These tests verify numerical stability of spectral computations using
small synthetic matrices. They are NOT used for benchmark evaluation.

All benchmark evidence comes from real agent traces only.
"""
import numpy as np
import pytest


def generate_synthetic_trace(seed: int, dim: int = 128, n_steps: int = 8) -> np.ndarray:
    """Generate a synthetic trace for testing (NOT for benchmarking).

    Parameters
    ----------
    seed : int
        Random seed for reproducibility
    dim : int
        Embedding dimension
    n_steps : int
        Number of steps

    Returns
    -------
    ndarray
        Synthetic trace matrix (n_steps x dim)
    """
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(dim)
    direction /= np.linalg.norm(direction)
    return np.array([
        (direction + rng.standard_normal(dim) * 0.1) / np.linalg.norm(direction + rng.standard_normal(dim) * 0.1)
        for _ in range(n_steps)
    ])


def compute_pca_energy(trace: np.ndarray, k: int = 3) -> float:
    """Compute PCA energy ratio (numerical regression test).

    Parameters
    ----------
    trace : ndarray
        Trace matrix
    k : int
        Number of top components

    Returns
    -------
    float
        Energy ratio
    """
    U, S, Vt = np.linalg.svd(trace, full_matrices=False)
    return float(np.sum(S[:k]**2) / (np.sum(S**2) + 1e-12))


def compute_koopman_residual(trace: np.ndarray) -> float:
    """Compute Koopman operator residual (numerical regression test).

    Parameters
    ----------
    trace : ndarray
        Trace matrix

    Returns
    -------
    float
        Normalized residual
    """
    X0, X1 = trace[:-1].T, trace[1:].T
    A = (X1 @ X0.T) @ np.linalg.pinv(X0 @ X0.T + 1e-12 * np.eye(X0.shape[0]))
    residual = np.linalg.norm(X1 - A @ X0, 'fro') / (np.linalg.norm(X1, 'fro') + 1e-12)
    return float(residual)


class TestSpectralMathRegression:
    """Regression tests for spectral math functions."""

    def test_pca_energy_stable_trajectory(self):
        """PCA energy should be high for low-rank trajectory."""
        trace = generate_synthetic_trace(seed=42, dim=64, n_steps=10)
        energy = compute_pca_energy(trace, k=3)

        # Low-rank synthetic trace should have high PCA energy
        assert energy > 0.7, f"Expected energy > 0.7, got {energy}"
        assert energy <= 1.0, f"Energy must be <= 1.0, got {energy}"

    def test_koopman_residual_stable_trajectory(self):
        """Koopman residual should be low for stable trajectory."""
        trace = generate_synthetic_trace(seed=42, dim=64, n_steps=10)
        residual = compute_koopman_residual(trace)

        # Stable trajectory should have low residual
        assert 0 <= residual <= 1.0, f"Residual out of range: {residual}"
        assert residual < 0.5, f"Expected low residual for stable trace, got {residual}"

    def test_spectral_bound_computation(self):
        """Spectral bound should combine PCA and Koopman correctly."""
        trace = generate_synthetic_trace(seed=100, dim=128, n_steps=15)

        pca_energy = compute_pca_energy(trace, k=3)
        koopman_residual = compute_koopman_residual(trace)
        spectral_bound = koopman_residual + (1 - pca_energy)

        # Verify bound properties
        assert 0 <= spectral_bound <= 2.0, f"Bound out of range: {spectral_bound}"
        assert spectral_bound == pytest.approx(koopman_residual + (1 - pca_energy))

    def test_numerical_stability_small_matrices(self):
        """Verify numerical stability on small matrices."""
        # Test edge case: minimal trace
        trace = generate_synthetic_trace(seed=1, dim=16, n_steps=3)

        pca_energy = compute_pca_energy(trace, k=2)
        koopman_residual = compute_koopman_residual(trace)

        # Should not raise errors and produce valid outputs
        assert np.isfinite(pca_energy)
        assert np.isfinite(koopman_residual)
        assert 0 <= pca_energy <= 1.0
        assert 0 <= koopman_residual <= 10.0  # Allow higher residual for minimal data

    def test_svd_reproducibility(self):
        """SVD should produce consistent results with same seed."""
        trace1 = generate_synthetic_trace(seed=12345, dim=64, n_steps=10)
        trace2 = generate_synthetic_trace(seed=12345, dim=64, n_steps=10)

        energy1 = compute_pca_energy(trace1, k=3)
        energy2 = compute_pca_energy(trace2, k=3)

        assert energy1 == pytest.approx(energy2, abs=1e-10)

    def test_drift_vs_gold_synthetic(self):
        """Synthetic drift trace should have higher bound than gold."""
        # Gold: stable trajectory
        gold = generate_synthetic_trace(seed=100, dim=128, n_steps=8)

        # Drift: add orthogonal component to simulate drift
        rng = np.random.default_rng(200)
        drift_direction = rng.standard_normal(128)
        drift_direction /= np.linalg.norm(drift_direction)

        drift = []
        for i in range(8):
            # Gradually drift away from gold direction
            mix = 0.3 + i * 0.1
            vec = (1 - mix) * gold[i] + mix * drift_direction
            drift.append(vec / np.linalg.norm(vec))
        drift = np.array(drift)

        gold_bound = compute_koopman_residual(gold) + (1 - compute_pca_energy(gold, k=3))
        drift_bound = compute_koopman_residual(drift) + (1 - compute_pca_energy(drift, k=3))

        # Drift should have higher bound (less stable)
        assert drift_bound > gold_bound, f"Drift bound {drift_bound} should exceed gold {gold_bound}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
