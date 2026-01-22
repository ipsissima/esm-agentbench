"""Numerical stability tests for spectral certificate computations."""
from __future__ import annotations

import numpy as np
import pytest

from certificates.make_certificate import compute_certificate

# Mark all tests in this module as unit tests (Tier 1)
pytestmark = pytest.mark.unit


def test_canonical_matrix_stability() -> None:
    X_canonical = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [1.0, 0.8, 0.6, 0.4, 0.2],
            [0.05, 0.1, 0.15, 0.2, 0.25],
            [0.9, 0.7, 0.5, 0.3, 0.1],
            [0.2, 0.4, 0.1, 0.3, 0.5],
        ],
        dtype=float,
    )

    cert = compute_certificate(X_canonical)

    expected_bound = 0.36661031527301335
    assert np.isclose(cert["theoretical_bound"], expected_bound, atol=1e-7)
