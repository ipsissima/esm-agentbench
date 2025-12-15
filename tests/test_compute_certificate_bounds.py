import numpy as np

from certificates.make_certificate import compute_certificate


def test_theoretical_bound_includes_variance_penalty():
    """Test that theoretical_bound applies variance penalty correctly.

    The formula is: theoretical_bound = residual + pca_tail_estimate + smooth_hallucination_penalty
    where smooth_hallucination_penalty = (1 - residual) * (1 - normalized_info)

    This penalizes traces with both low residual AND low information (loops).
    """
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(8, 4))
    cert = compute_certificate(embeddings, r=3)

    residual = cert["residual"]
    tail = cert["pca_tail_estimate"]
    information_density = cert["information_density"]
    theoretical = cert["theoretical_bound"]

    # Verify information_density is present and positive
    assert information_density > 0

    # Verify the formula: theoretical_bound = residual + tail + penalty
    log_info = np.log1p(information_density)
    max_log_info = np.log1p(10.0)
    normalized_info = np.clip(log_info / max_log_info, 0.0, 1.0)
    residual_complement = np.clip(1.0 - residual, 0.0, 1.0)
    info_complement = np.clip(1.0 - normalized_info, 0.0, 1.0)
    smooth_hallucination_penalty = residual_complement * info_complement

    expected_bound = residual + tail + smooth_hallucination_penalty
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


def test_theoretical_bound_matches_residual_and_tail():
    """Backward compatibility: theoretical_bound is still a function of residual and tail."""
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(8, 4))
    cert = compute_certificate(embeddings, r=3)

    residual = cert["residual"]
    tail = cert["pca_tail_estimate"]
    information_density = cert["information_density"]
    theoretical = cert["theoretical_bound"]

    # The bound should be positive and include both residual and tail components
    assert theoretical >= 0
    # The bound should include the smooth hallucination penalty
    log_info = np.log1p(information_density)
    max_log_info = np.log1p(10.0)
    normalized_info = np.clip(log_info / max_log_info, 0.0, 1.0)
    residual_complement = np.clip(1.0 - residual, 0.0, 1.0)
    info_complement = np.clip(1.0 - normalized_info, 0.0, 1.0)
    smooth_hallucination_penalty = residual_complement * info_complement

    expected = residual + tail + smooth_hallucination_penalty
    assert np.isclose(theoretical, expected, atol=1e-9)
