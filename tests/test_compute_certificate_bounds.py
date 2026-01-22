"""Tests for SVD-based spectral certificate bounds.

The new certificate uses Wedin's Theorem and SVD for rigorous bounds:
    theoretical_bound = C_res * residual + C_tail * tail_energy + C_sem * semantic_divergence

Where C_res, C_tail, and C_sem are constants from formal verification (Coq/UELAT).
The semantic_divergence term enables detection of poison/adversarial attacks.
"""
import numpy as np
import pytest

from certificates.make_certificate import compute_certificate

# Mark all tests in this module as unit tests (Tier 1)
pytestmark = pytest.mark.unit


def test_theoretical_bound_matches_formula():
    """Verify theoretical_bound matches the rigorous SVD-based formula.

    The formula is:
        theoretical_bound = C_res * residual + C_tail * tail_energy + C_sem * semantic_divergence

    Where C_res, C_tail, and C_sem are verified constants.
    """
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(12, 8))
    cert = compute_certificate(embeddings, r=3)

    residual = cert["residual"]
    tail_energy = cert["tail_energy"]
    semantic_divergence = cert["semantic_divergence"]
    theoretical = cert["theoretical_bound"]
    c_res = cert["C_res"]
    c_tail = cert["C_tail"]
    c_sem = cert["C_sem"]

    # Verify the formula matches (now includes semantic divergence)
    expected_bound = c_res * residual + c_tail * tail_energy + c_sem * semantic_divergence
    assert np.isclose(theoretical, expected_bound, atol=1e-9)


def test_low_variance_trace_has_high_pca_explained():
    """Test that low-variance (loop-like) traces have high PCA explained."""
    rng = np.random.default_rng(42)
    base_embedding = rng.normal(size=(1, 64))
    # Repeat with tiny noise - simulates identity mapping/loop
    loop_trace = np.tile(base_embedding, (10, 1)) + rng.normal(size=(10, 64)) * 0.001

    # Create a high-variance (progressive) trace
    progressive_trace = rng.normal(size=(10, 64))

    cert_loop = compute_certificate(loop_trace, r=3)
    cert_progressive = compute_certificate(progressive_trace, r=3)

    # Loop trace should have very high PCA explained (nearly all variance in first component)
    assert cert_loop["pca_explained"] > 0.99

    # Progressive trace should have lower PCA explained (variance spread across components)
    assert cert_progressive["pca_explained"] < cert_loop["pca_explained"]


def test_koopman_singular_values_present():
    """Test that Koopman operator singular values are correctly computed."""
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(12, 64))
    cert = compute_certificate(embeddings, r=3)

    # Verify Koopman singular value fields are present
    assert "koopman_sigma_max" in cert
    assert "koopman_sigma_second" in cert
    assert "koopman_singular_gap" in cert

    # Leading singular value should be larger than second
    assert cert["koopman_sigma_max"] >= cert["koopman_sigma_second"]

    # Singular gap should match the difference
    expected_gap = cert["koopman_sigma_max"] - cert["koopman_sigma_second"]
    assert np.isclose(cert["koopman_singular_gap"], expected_gap, atol=1e-9)


def test_tail_energy_and_pca_explained_sum_to_one():
    """Test that tail_energy + pca_explained ≈ 1."""
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(15, 32))
    cert = compute_certificate(embeddings, r=3)

    # tail_energy is 1 - pca_explained
    total = cert["pca_explained"] + cert["tail_energy"]
    assert np.isclose(total, 1.0, atol=1e-9)


def test_verified_constants_present():
    """Test that C_res and C_tail constants from formal verification are included."""
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(10, 8))
    cert = compute_certificate(embeddings, r=3)

    # Constants should be present for transparency
    assert "C_res" in cert
    assert "C_tail" in cert

    # Constants should be positive
    assert cert["C_res"] > 0
    assert cert["C_tail"] > 0


def test_singular_values_of_data_matrix():
    """Test that data matrix singular values are correctly computed."""
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(12, 8))
    cert = compute_certificate(embeddings, r=3)

    # Verify singular value fields are present
    assert "sigma_max" in cert
    assert "sigma_second" in cert
    assert "singular_gap" in cert

    # Verify ordering
    assert cert["sigma_max"] >= cert["sigma_second"]


def test_residual_is_normalized():
    """Test that residual is properly normalized between 0 and 1."""
    rng = np.random.default_rng(42)

    # Test with well-structured data (should have low residual)
    X = rng.normal(size=(20, 10))
    cert = compute_certificate(X, r=5)
    assert 0.0 <= cert["residual"] <= 1.0

    # Test with random data
    Y = rng.normal(size=(50, 30))
    cert2 = compute_certificate(Y, r=10)
    assert 0.0 <= cert2["residual"] <= 1.0
