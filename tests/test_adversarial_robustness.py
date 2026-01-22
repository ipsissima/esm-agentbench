"""Adversarial robustness tests for the spectral certificate metric."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.linalg import norm

from certificates.make_certificate import compute_certificate

# Mark all tests in this module as unit tests (Tier 1)
pytestmark = pytest.mark.unit


def _create_task_embedding(dim: int = 64, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    task = rng.standard_normal(dim)
    return task / (norm(task) + 1e-12)


def _orthogonal_unit_vector(task_embedding: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    vec = rng.standard_normal(len(task_embedding))
    vec = vec - np.dot(vec, task_embedding) * task_embedding
    return vec / (norm(vec) + 1e-12)


def _create_orthogonal_trace(
    task_embedding: np.ndarray, n_steps: int = 16, seed: int = 123
) -> np.ndarray:
    """Generate a trace that stays orthogonal to the task vector."""
    rng = np.random.default_rng(seed)
    embeddings = []
    for _ in range(n_steps):
        vec = _orthogonal_unit_vector(task_embedding, rng)
        embeddings.append(vec)
    return np.array(embeddings)


def _create_base_trace(
    task_embedding: np.ndarray, n_steps: int = 16, seed: int = 456
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(len(task_embedding)) * 0.1
    direction = direction / (norm(direction) + 1e-12)
    embeddings = []
    for step in range(n_steps):
        vec = task_embedding + direction * step * 0.05
        vec = vec / (norm(vec) + 1e-12)
        embeddings.append(vec)
    return np.array(embeddings)


def test_orthogonal_attack() -> None:
    task_embedding = _create_task_embedding(dim=64, seed=10)
    drift_trace = _create_orthogonal_trace(task_embedding, seed=11)

    cert = compute_certificate(drift_trace, task_embedding=task_embedding)

    assert cert["theoretical_bound"] > 0.8


def test_scaling_invariance() -> None:
    task_embedding = _create_task_embedding(dim=64, seed=30)
    base_trace = _create_base_trace(task_embedding, seed=31)

    base_cert = compute_certificate(base_trace, task_embedding=task_embedding)
    scaled_cert = compute_certificate(base_trace * 1e6, task_embedding=task_embedding)

    assert np.isclose(base_cert["residual"], scaled_cert["residual"], rtol=1e-6, atol=1e-6)
    assert np.isclose(
        base_cert["theoretical_bound"], scaled_cert["theoretical_bound"], rtol=1e-6, atol=1e-6
    )


def test_noise_sensitivity() -> None:
    task_embedding = _create_task_embedding(dim=64, seed=40)
    base_trace = _create_base_trace(task_embedding, seed=41)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(base_trace.shape)

    bounds = []
    for scale in [0.0, 0.05, 0.1]:
        noisy_trace = base_trace + scale * noise
        cert = compute_certificate(noisy_trace, task_embedding=task_embedding)
        bounds.append(cert["theoretical_bound"])

    assert bounds[0] <= bounds[1] <= bounds[2]
