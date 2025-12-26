"""Pytest configuration and fixtures for ESM-AgentBench tests.

This module provides test fixtures and mocks for running tests without
the verified kernel (which requires Coq/OCaml build).
"""
import os
from unittest.mock import patch, MagicMock

import pytest
import numpy as np


@pytest.fixture(autouse=True)
def mock_verified_kernel():
    """Mock the verified kernel for all tests.

    The verified kernel requires a compiled Coq/OCaml library that may not
    be available in all test environments. This fixture provides a Python
    fallback implementation for testing purposes.
    """
    def mock_kernel_compute_certificate(X0, X1, A, tail_energy, semantic_divergence, lipschitz_margin, strict=True):
        """Fallback Python implementation of kernel certificate computation.

        This is a pure Python implementation that mirrors the verified kernel's
        logic for testing purposes. In production, the verified kernel provides
        formal guarantees.
        """
        eps = 1e-12

        # Compute residual: ||X1 - A @ X0||_F / ||X1||_F
        err = X1 - A @ X0
        residual = float(np.linalg.norm(err, "fro") / (np.linalg.norm(X1, "fro") + eps))

        # Compute bound using constants
        c_res = 1.0
        c_tail = 1.0
        c_sem = 1.0
        c_robust = 1.0

        theoretical_bound = float(
            c_res * residual +
            c_tail * tail_energy +
            c_sem * semantic_divergence +
            c_robust * lipschitz_margin
        )

        return (residual, theoretical_bound)

    # Patch the kernel's compute_certificate function
    with patch('certificates.verified_kernel.compute_certificate', side_effect=mock_kernel_compute_certificate):
        # Also patch the import in make_certificate
        with patch('certificates.make_certificate.kernel_compute_certificate', side_effect=mock_kernel_compute_certificate):
            yield


@pytest.fixture
def rng():
    """Provide a seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_embeddings(rng):
    """Generate sample embeddings for testing."""
    return rng.normal(size=(30, 64))


@pytest.fixture
def sample_task_embedding(rng):
    """Generate a sample task embedding."""
    return rng.normal(size=64)
