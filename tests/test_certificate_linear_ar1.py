import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from certificates.make_certificate import compute_certificate


def test_ar1_low_residual_and_correct_max_eig():
    rng = np.random.default_rng(0)
    B = rng.standard_normal((3, 3)) * 0.5
    eigs = np.linalg.eigvals(B)
    true_max = np.max(np.abs(eigs))

    x = rng.standard_normal(3)
    X = []
    for _ in range(20):
        X.append(x.copy())
        x = B @ x
    X = np.array(X)

    cert = compute_certificate(X, r=3)

    assert cert["residual"] < 1e-6
    assert abs(cert["max_eig"] - true_max) < 1e-3


def test_short_sequence_defaults():
    cert = compute_certificate([])
    assert cert["residual"] == 1.0
    assert cert["spectral_gap"] == 0.0
