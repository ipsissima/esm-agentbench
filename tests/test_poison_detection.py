"""Unit tests for poison detection via semantic divergence.

This test validates that the spectral certificate correctly identifies
"stable but wrong direction" traces (poison/adversarial attacks) that would
previously pass as "Gold" traces due to low residual/high stability.

The key insight is:
- Poison traces are STABLE (low residual, predictable dynamics)
- But semantically WRONG (orthogonal to task intent)

Before semantic divergence: Both Gold and Poison would have low theoretical_bound
After semantic divergence: Gold stays low, Poison is penalized and flagged high
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import norm
import pytest

from certificates.make_certificate import compute_certificate


def _create_task_embedding(dim: int = 64, seed: int = 42) -> np.ndarray:
    """Create a normalized task embedding."""
    rng = np.random.default_rng(seed)
    task = rng.standard_normal(dim)
    return task / (norm(task) + 1e-12)


def _create_gold_trace(
    task_embedding: np.ndarray, n_steps: int = 12, seed: int = 42
) -> np.ndarray:
    """Create a coherent trace that stays aligned with the task.

    This simulates a model correctly following the user's intent.
    Should have: LOW residual, LOW semantic divergence -> LOW bound
    """
    dim = len(task_embedding)
    rng = np.random.default_rng(seed)

    # Small direction for smooth evolution
    direction = rng.standard_normal(dim) * 0.1
    direction = direction / (norm(direction) + 1e-12)

    embeddings = []
    for step in range(n_steps):
        # Stay close to task with smooth evolution
        vec = task_embedding + direction * step * 0.1 + rng.standard_normal(dim) * 0.05
        vec = vec / (norm(vec) + 1e-12)
        embeddings.append(vec)

    return np.array(embeddings)


def _create_stable_orthogonal_trace(
    task_embedding: np.ndarray, n_steps: int = 12, seed: int = 42
) -> np.ndarray:
    """Create a perfectly stable trace that is ORTHOGONAL to the task.

    This is the "silent failure" scenario:
    - The trace is STABLE (low residual, predictable dynamics)
    - But semantically WRONG (orthogonal to task intent)

    This simulates a jailbreak or prompt injection where the model
    confidently outputs off-topic content.

    Should have: LOW residual, HIGH semantic divergence -> HIGH bound
    """
    dim = len(task_embedding)
    rng = np.random.default_rng(seed)

    # Create a base that is orthogonal to task (completely different semantic space)
    poison_base = rng.standard_normal(dim)
    # Gram-Schmidt: remove component along task direction
    poison_base = poison_base - np.dot(poison_base, task_embedding) * task_embedding
    poison_base = poison_base / (norm(poison_base) + 1e-12)

    # Create a direction that is also orthogonal to task (stay in orthogonal subspace)
    direction = rng.standard_normal(dim)
    direction = direction - np.dot(direction, task_embedding) * task_embedding
    direction = direction / (norm(direction) + 1e-12)

    embeddings = []
    for step in range(n_steps):
        # Smooth, confident evolution in the orthogonal subspace
        # Each vector is constructed to be orthogonal to task
        vec = poison_base + direction * step * 0.03
        # Re-orthogonalize to ensure we stay orthogonal to task
        vec = vec - np.dot(vec, task_embedding) * task_embedding
        # Add very small noise that doesn't break orthogonality
        noise = rng.standard_normal(dim) * 0.01
        noise = noise - np.dot(noise, task_embedding) * task_embedding
        vec = vec + noise
        vec = vec / (norm(vec) + 1e-12)
        embeddings.append(vec)

    return np.array(embeddings)


class TestPoisonDetection:
    """Test suite for poison detection via semantic divergence."""

    def test_stable_orthogonal_trace_is_flagged(self):
        """A perfectly stable trace orthogonal to task should have HIGH bound.

        This is the core test for the semantic divergence feature.
        Previously, this would have passed as "Gold" due to low residual.
        Now, it must be flagged due to high semantic divergence.
        """
        task_embedding = _create_task_embedding(dim=64, seed=100)
        poison_trace = _create_stable_orthogonal_trace(task_embedding, seed=100)

        cert = compute_certificate(poison_trace, task_embedding=task_embedding)

        # Semantic divergence should be HIGH (orthogonal = cosine distance ~1.0)
        assert cert["semantic_divergence"] > 0.8, (
            f"Orthogonal trace should have high semantic divergence, "
            f"got {cert['semantic_divergence']}"
        )

        # The theoretical bound should be HIGH (flagged)
        # With C_sem = 1.0 and divergence ~1.0, this adds ~1.0 to the bound
        assert cert["theoretical_bound"] > 1.0, (
            f"Stable orthogonal trace should have high bound due to semantic term, "
            f"got {cert['theoretical_bound']}"
        )

    def test_gold_trace_has_low_bound(self):
        """A coherent on-task trace should have LOW bound."""
        task_embedding = _create_task_embedding(dim=64, seed=200)
        gold_trace = _create_gold_trace(task_embedding, seed=200)

        cert = compute_certificate(gold_trace, task_embedding=task_embedding)

        # Semantic divergence should be LOW (aligned with task)
        assert cert["semantic_divergence"] < 0.5, (
            f"Gold trace should have low semantic divergence, "
            f"got {cert['semantic_divergence']}"
        )

        # The theoretical bound should be LOW (not flagged)
        assert cert["theoretical_bound"] < 1.0, (
            f"Gold trace should have low bound, got {cert['theoretical_bound']}"
        )

    def test_poison_bound_higher_than_gold(self):
        """Poison traces must have significantly higher bounds than gold traces.

        This is the key discrimination test. The semantic divergence term
        should push poison traces above gold traces.
        """
        task_embedding = _create_task_embedding(dim=64, seed=300)

        gold_trace = _create_gold_trace(task_embedding, seed=300)
        poison_trace = _create_stable_orthogonal_trace(task_embedding, seed=301)

        gold_cert = compute_certificate(gold_trace, task_embedding=task_embedding)
        poison_cert = compute_certificate(poison_trace, task_embedding=task_embedding)

        # Poison should have higher bound than gold
        assert poison_cert["theoretical_bound"] > gold_cert["theoretical_bound"], (
            f"Poison bound ({poison_cert['theoretical_bound']:.4f}) should be > "
            f"Gold bound ({gold_cert['theoretical_bound']:.4f})"
        )

        # The difference should be significant (at least 0.5)
        diff = poison_cert["theoretical_bound"] - gold_cert["theoretical_bound"]
        assert diff > 0.5, (
            f"Difference between poison and gold bounds should be significant, "
            f"got {diff:.4f}"
        )

    def test_poison_has_low_residual_but_high_divergence(self):
        """Verify that poison traces are stable (low residual) but divergent.

        This validates the core assumption: poison traces are "confident but wrong".
        """
        task_embedding = _create_task_embedding(dim=64, seed=400)
        poison_trace = _create_stable_orthogonal_trace(task_embedding, seed=400)

        cert = compute_certificate(poison_trace, task_embedding=task_embedding)

        # Residual should be LOW (stable dynamics)
        assert cert["residual"] < 0.3, (
            f"Poison trace should have low residual (stable), got {cert['residual']}"
        )

        # But semantic divergence should be HIGH
        assert cert["semantic_divergence"] > 0.8, (
            f"Poison trace should have high semantic divergence, "
            f"got {cert['semantic_divergence']}"
        )

    def test_no_task_embedding_uses_first_step(self):
        """When no task embedding provided, first step is used as reference.

        This is a fallback behavior test.
        """
        task_embedding = _create_task_embedding(dim=64, seed=500)
        gold_trace = _create_gold_trace(task_embedding, seed=500)

        # Call without task_embedding
        cert = compute_certificate(gold_trace, task_embedding=None)

        # Should still compute a valid certificate
        assert "semantic_divergence" in cert
        assert "theoretical_bound" in cert
        # Semantic divergence should still be relatively low since trace evolves smoothly
        assert cert["semantic_divergence"] < 0.8

    def test_statistical_separation_multiple_runs(self):
        """Run multiple random seeds to verify statistical separation."""
        gold_bounds = []
        poison_bounds = []

        for seed in range(10):
            task_embedding = _create_task_embedding(dim=64, seed=seed * 1000)

            gold_trace = _create_gold_trace(task_embedding, seed=seed * 1000 + 1)
            poison_trace = _create_stable_orthogonal_trace(task_embedding, seed=seed * 1000 + 2)

            gold_cert = compute_certificate(gold_trace, task_embedding=task_embedding)
            poison_cert = compute_certificate(poison_trace, task_embedding=task_embedding)

            gold_bounds.append(gold_cert["theoretical_bound"])
            poison_bounds.append(poison_cert["theoretical_bound"])

        gold_mean = np.mean(gold_bounds)
        poison_mean = np.mean(poison_bounds)

        # Poison mean should be significantly higher than gold mean
        assert poison_mean > gold_mean, (
            f"Poison mean ({poison_mean:.4f}) should be > Gold mean ({gold_mean:.4f})"
        )

        # The effect size should be large
        effect_size = (poison_mean - gold_mean) / np.std(gold_bounds + poison_bounds)
        assert effect_size > 1.0, (
            f"Effect size should be large (Cohen's d > 1.0), got {effect_size:.4f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
