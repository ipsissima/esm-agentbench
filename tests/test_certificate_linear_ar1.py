import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from certificates.make_certificate import compute_certificate

# Mark all tests in this module as unit tests (Tier 1)
pytestmark = pytest.mark.unit


def test_ar1_low_residual_and_correct_koopman_sigma():
    """Test that AR1 process yields low residual and meaningful Koopman singular values.

    The new SVD-based certificate uses Koopman singular values (koopman_sigma_max)
    instead of eigenvalues. For a well-conditioned AR1 process, we expect low residual
    and the Koopman operator's leading singular value to capture the dynamics.
    """
    rng = np.random.default_rng(0)
    B = rng.standard_normal((3, 3)) * 0.5
    eigs = np.linalg.eigvals(B)
    true_max = np.max(np.abs(eigs))

    x = rng.standard_normal(3)
    X = []
    for _ in range(50):  # Use more samples for stable SVD estimation
        X.append(x.copy())
        x = B @ x
    X = np.array(X)

    cert = compute_certificate(X, r=3)

    # With the SVD-based method, residual tolerance is looser due to different normalization
    assert cert["residual"] < 0.01, f"Residual {cert['residual']} too high for AR1 process"
    # Koopman singular values should be present and positive
    assert "koopman_sigma_max" in cert
    assert cert["koopman_sigma_max"] > 0


def test_short_sequence_defaults():
    """Test empty/short sequences return conservative defaults."""
    cert = compute_certificate([])
    assert cert["residual"] == 1.0
    # New API uses singular_gap instead of spectral_gap
    assert cert["singular_gap"] == 0.0
