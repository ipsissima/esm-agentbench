import numpy as np

from certificates.make_certificate import compute_certificate


def test_theoretical_bound_matches_residual_and_tail():
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(8, 4))
    cert = compute_certificate(embeddings, r=3)

    residual = cert["residual"]
    tail = 1.0 - cert["pca_explained"]
    theoretical = cert["theoretical_bound"]

    assert theoretical >= residual
    assert np.isclose(theoretical, residual + tail, atol=1e-9)
