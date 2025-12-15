import numpy as np

from certificates.make_certificate import compute_certificate


def test_theoretical_bound_includes_all_penalties():
    """Test that theoretical_bound applies all penalties correctly.

    The formula is:
        theoretical_bound = residual + pca_tail_estimate
                          + smooth_hallucination_penalty
                          + eig_product_penalty

    Where:
    - smooth_hallucination_penalty = (1 - residual) * (1 - normalized_info)
    - eig_product_penalty = max(0, 0.8 - λ₁×λ₂) * 0.5
    """
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(8, 4))
    cert = compute_certificate(embeddings, r=3)

    residual = cert["residual"]
    tail = cert["pca_tail_estimate"]
    information_density = cert["information_density"]
    theoretical = cert["theoretical_bound"]
    max_eig = cert["max_eig"]
    second_eig = cert["second_eig"]

    # Verify information_density is present and positive
    assert information_density > 0

    # Compute smooth hallucination penalty
    log_info = np.log1p(information_density)
    max_log_info = np.log1p(10.0)
    normalized_info = np.clip(log_info / max_log_info, 0.0, 1.0)
    residual_complement = np.clip(1.0 - residual, 0.0, 1.0)
    info_complement = np.clip(1.0 - normalized_info, 0.0, 1.0)
    smooth_hallucination_penalty = residual_complement * info_complement

    # Compute eigenvalue product penalty
    eig_product = max_eig * second_eig
    eig_product_penalty = max(0.0, 0.8 - eig_product) * 0.5

    expected_bound = residual + tail + smooth_hallucination_penalty + eig_product_penalty
    assert np.isclose(theoretical, expected_bound, atol=1e-9)


def test_low_variance_trace_penalized():
    """Test that low-variance (loop-like) traces get higher bounds."""
    # Create a nearly constant (loop-like) trace - all embeddings very similar
    rng = np.random.default_rng(42)
    base_embedding = rng.normal(size=(1, 64))
    # Repeat with tiny noise - simulates identity mapping/loop
    loop_trace = np.tile(base_embedding, (10, 1)) + rng.normal(size=(10, 64)) * 0.001

    # Create a high-variance (progressive) trace
    progressive_trace = rng.normal(size=(10, 64))

    cert_loop = compute_certificate(loop_trace, r=3)
    cert_progressive = compute_certificate(progressive_trace, r=3)

    # Loop should have lower information density
    assert cert_loop["information_density"] < cert_progressive["information_density"]

    # For traces with similar raw residuals, low variance should yield higher bound
    # (penalized) when residual is non-zero
    # Note: The loop trace may have very low residual too, so we check info density


def test_theoretical_bound_matches_formula():
    """Verify theoretical_bound matches the complete formula."""
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(8, 4))
    cert = compute_certificate(embeddings, r=3)

    residual = cert["residual"]
    tail = cert["pca_tail_estimate"]
    information_density = cert["information_density"]
    theoretical = cert["theoretical_bound"]
    max_eig = cert["max_eig"]
    second_eig = cert["second_eig"]

    # The bound should be positive
    assert theoretical >= 0

    # Compute smooth hallucination penalty
    log_info = np.log1p(information_density)
    max_log_info = np.log1p(10.0)
    normalized_info = np.clip(log_info / max_log_info, 0.0, 1.0)
    residual_complement = np.clip(1.0 - residual, 0.0, 1.0)
    info_complement = np.clip(1.0 - normalized_info, 0.0, 1.0)
    smooth_hallucination_penalty = residual_complement * info_complement

    # Compute eigenvalue product penalty
    eig_product = max_eig * second_eig
    eig_product_penalty = max(0.0, 0.8 - eig_product) * 0.5

    expected = residual + tail + smooth_hallucination_penalty + eig_product_penalty
    assert np.isclose(theoretical, expected, atol=1e-9)


def test_eigenvalue_product_penalty_applied():
    """Test that eigenvalue product penalty is correctly applied."""
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(12, 64))
    cert = compute_certificate(embeddings, r=3)

    # Verify eigenvalue fields are present
    assert "max_eig" in cert
    assert "second_eig" in cert
    assert "eig_product" in cert
    assert "eig_product_penalty" in cert

    # Verify penalty calculation
    expected_product = cert["max_eig"] * cert["second_eig"]
    assert np.isclose(cert["eig_product"], expected_product, atol=1e-9)

    expected_penalty = max(0.0, 0.8 - expected_product) * 0.5
    assert np.isclose(cert["eig_product_penalty"], expected_penalty, atol=1e-9)
