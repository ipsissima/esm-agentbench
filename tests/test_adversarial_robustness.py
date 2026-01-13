"""Adversarial robustness tests for the spectral certificate metric.

These tests stress the semantic divergence term with targeted attacks:
- Orthogonal drift (confident but wrong direction)
- High-frequency noise within a low-rank subspace
- Scaling invariance of normalized embeddings
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import norm
import pytest

from certificates.make_certificate import compute_certificate


def _create_task_embedding(dim: int = 64, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    task = rng.standard_normal(dim)
    return task / (norm(task) + 1e-12)


def _orthogonal_unit_vector(task_embedding: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    vec = rng.standard_normal(len(task_embedding))
    vec = vec - np.dot(vec, task_embedding) * task_embedding
    return vec / (norm(vec) + 1e-12)


def _create_orthogonal_drift_trace(
    task_embedding: np.ndarray, n_steps: int = 16, seed: int = 123
) -> np.ndarray:
    """Generate a trace that drifts orthogonally to the task vector."""
    rng = np.random.default_rng(seed)
    base = _orthogonal_unit_vector(task_embedding, rng)
    direction = _orthogonal_unit_vector(task_embedding, rng)

    embeddings = []
    for step in range(n_steps):
        vec = base + direction * step * 0.05
        vec = vec - np.dot(vec, task_embedding) * task_embedding
        vec = vec / (norm(vec) + 1e-12)
        embeddings.append(vec)
    return np.array(embeddings)


def _create_high_frequency_noise_trace(
    task_embedding: np.ndarray, n_steps: int = 16, seed: int = 321
) -> np.ndarray:
    """Inject high-frequency noise while staying in a low-rank orthogonal subspace."""
    rng = np.random.default_rng(seed)
    base = _orthogonal_unit_vector(task_embedding, rng)
    noise_dir = _orthogonal_unit_vector(task_embedding, rng)

    embeddings = []
    for step in range(n_steps):
        sign = -1.0 if step % 2 else 1.0
        vec = base + sign * 0.6 * noise_dir
        vec = vec - np.dot(vec, task_embedding) * task_embedding
        vec = vec / (norm(vec) + 1e-12)
        embeddings.append(vec)
    return np.array(embeddings)


def _create_gold_trace(
    task_embedding: np.ndarray, n_steps: int = 16, seed: int = 456
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(len(task_embedding)) * 0.1
    direction = direction / (norm(direction) + 1e-12)
    embeddings = []
    for step in range(n_steps):
        vec = task_embedding + direction * step * 0.08 + rng.standard_normal(len(task_embedding)) * 0.03
        vec = vec / (norm(vec) + 1e-12)
        embeddings.append(vec)
    return np.array(embeddings)


class TestAdversarialRobustness:
    """Adversarial robustness harness for the spectral certificate metric."""

    def test_orthogonal_drift_attack_flagged(self):
        task_embedding = _create_task_embedding(dim=64, seed=10)
        drift_trace = _create_orthogonal_drift_trace(task_embedding, seed=11)

        cert = compute_certificate(drift_trace, task_embedding=task_embedding)

        assert cert["semantic_divergence"] > 0.8, (
            f"Orthogonal drift should have high semantic divergence, got {cert['semantic_divergence']}"
        )
        assert cert["theoretical_bound"] > 1.0, (
            f"Orthogonal drift should have high bound, got {cert['theoretical_bound']}"
        )

    def test_high_frequency_noise_attack_flagged(self):
        task_embedding = _create_task_embedding(dim=64, seed=20)
        noisy_trace = _create_high_frequency_noise_trace(task_embedding, seed=21)

        cert = compute_certificate(noisy_trace, task_embedding=task_embedding)

        assert cert["semantic_divergence"] > 0.8, (
            f"High-frequency noise should have high semantic divergence, got {cert['semantic_divergence']}"
        )
        assert cert["theoretical_bound"] > 1.0, (
            f"High-frequency noise should have high bound, got {cert['theoretical_bound']}"
        )

    @pytest.mark.parametrize("scale", [0.1, 10.0, 1000.0])
    def test_scaling_attack_invariant(self, scale: float):
        task_embedding = _create_task_embedding(dim=64, seed=30)
        gold_trace = _create_gold_trace(task_embedding, seed=31)

        base_cert = compute_certificate(gold_trace, task_embedding=task_embedding)
        scaled_cert = compute_certificate(gold_trace * scale, task_embedding=task_embedding)

        assert np.isclose(
            base_cert["theoretical_bound"], scaled_cert["theoretical_bound"], rtol=1e-6, atol=1e-6
        ), (
            "Theoretical bound should be invariant to global scaling, "
            f"got base={base_cert['theoretical_bound']} scaled={scaled_cert['theoretical_bound']}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
