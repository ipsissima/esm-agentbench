"""Tests for SVD sign canonicalization and reproducibility."""
from __future__ import annotations

import hashlib
import numpy as np
import pytest

from certificates.make_certificate import (
    _compute_svd,
    _canonicalize_svd_signs,
    _encode_matrix_base64,
    _hash_matrix_bytes,
)

# Mark all tests in this module as unit tests (Tier 1)
pytestmark = pytest.mark.unit


def test_canonicalize_svd_signs_basic() -> None:
    """Test that sign canonicalization enforces positive largest absolute entry."""
    np.random.seed(42)
    X = np.random.randn(5, 3)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Canonicalize
    U_canon, S_canon, Vt_canon = _canonicalize_svd_signs(U, S, Vt)
    
    # Verify singular values unchanged
    np.testing.assert_array_almost_equal(S, S_canon)
    
    # Verify each column of U has positive largest absolute entry
    for i in range(U_canon.shape[1]):
        max_idx = np.argmax(np.abs(U_canon[:, i]))
        assert U_canon[max_idx, i] >= 0, f"Column {i} has negative largest entry"
    
    # Verify reconstruction is the same
    X_reconstructed = U_canon @ np.diag(S_canon) @ Vt_canon
    np.testing.assert_array_almost_equal(X, X_reconstructed, decimal=10)


def test_canonicalize_svd_signs_stability() -> None:
    """Test that canonicalization produces stable results across multiple runs."""
    np.random.seed(123)
    X = np.random.randn(10, 8)
    
    # Compute SVD multiple times (different random states for comparison)
    results = []
    for seed in range(5):
        np.random.seed(seed)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        U_canon, S_canon, Vt_canon = _canonicalize_svd_signs(U, S, Vt)
        results.append((U_canon, S_canon, Vt_canon))
    
    # All canonicalized results should be identical (up to numerical precision)
    # Note: This test assumes np.linalg.svd is deterministic for the same input
    for i in range(1, len(results)):
        np.testing.assert_array_almost_equal(results[0][0], results[i][0], decimal=10)
        np.testing.assert_array_almost_equal(results[0][1], results[i][1], decimal=10)
        np.testing.assert_array_almost_equal(results[0][2], results[i][2], decimal=10)


def test_compute_svd_with_canonicalization() -> None:
    """Test that _compute_svd applies canonicalization by default."""
    np.random.seed(456)
    X = np.random.randn(6, 4)
    
    # Compute with canonicalization (default)
    U_canon, S_canon, Vt_canon = _compute_svd(X, r=4, canonicalize_signs=True)
    
    # Verify each column of U has positive largest absolute entry
    for i in range(U_canon.shape[1]):
        max_idx = np.argmax(np.abs(U_canon[:, i]))
        assert U_canon[max_idx, i] >= 0, f"Column {i} has negative largest entry"
    
    # Verify reconstruction
    X_reconstructed = U_canon @ np.diag(S_canon) @ Vt_canon
    np.testing.assert_array_almost_equal(X, X_reconstructed, decimal=10)


def test_svd_hash_stability() -> None:
    """Test that canonicalized SVD produces stable SHA256 hashes."""
    np.random.seed(789)
    X = np.random.randn(8, 6)
    
    # Compute SVD with canonicalization multiple times
    hashes = []
    for _ in range(3):
        U, S, Vt = _compute_svd(X, r=6, canonicalize_signs=True)
        # Hash the concatenated matrices
        hash_u = _hash_matrix_bytes(U)
        hash_vt = _hash_matrix_bytes(Vt)
        combined_hash = hashlib.sha256((hash_u + hash_vt).encode()).hexdigest()
        hashes.append(combined_hash)
    
    # All hashes should be identical
    assert len(set(hashes)) == 1, "SVD hashes are not stable across runs"


def test_svd_without_canonicalization() -> None:
    """Test that canonicalization can be disabled."""
    np.random.seed(101)
    X = np.random.randn(5, 4)
    
    # Compute without canonicalization
    U, S, Vt = _compute_svd(X, r=4, canonicalize_signs=False)
    
    # Just verify reconstruction works (signs may vary)
    X_reconstructed = U @ np.diag(S) @ Vt
    np.testing.assert_array_almost_equal(X, X_reconstructed, decimal=10)


def test_randomized_svd_canonicalization() -> None:
    """Test that canonicalization works with randomized SVD."""
    np.random.seed(202)
    X = np.random.randn(20, 15)
    
    # Compute with randomized SVD and canonicalization
    U_rand, S_rand, Vt_rand = _compute_svd(
        X, r=10, use_randomized_svd=True, 
        randomized_svd_random_state=42, 
        canonicalize_signs=True
    )
    
    # Verify each column of U has positive largest absolute entry
    for i in range(U_rand.shape[1]):
        max_idx = np.argmax(np.abs(U_rand[:, i]))
        assert U_rand[max_idx, i] >= 0, f"Column {i} has negative largest entry"
    
    # Verify reconstruction quality (randomized SVD is approximate)
    X_reconstructed = U_rand @ np.diag(S_rand) @ Vt_rand
    relative_error = np.linalg.norm(X - X_reconstructed, 'fro') / np.linalg.norm(X, 'fro')
    assert relative_error < 0.01, f"Randomized SVD reconstruction error too large: {relative_error}"


def test_encode_matrix_stable_after_canonicalization() -> None:
    """Test that encoded matrices have stable base64 strings after canonicalization."""
    np.random.seed(303)
    X = np.random.randn(7, 5)
    
    # Compute SVD with canonicalization multiple times
    encodings = []
    for _ in range(3):
        U, S, Vt = _compute_svd(X, r=5, canonicalize_signs=True)
        encoded_u = _encode_matrix_base64(U)
        encoded_vt = _encode_matrix_base64(Vt)
        encodings.append((encoded_u, encoded_vt))
    
    # All encodings should be identical
    assert len(set(enc[0] for enc in encodings)) == 1, "U encodings not stable"
    assert len(set(enc[1] for enc in encodings)) == 1, "Vt encodings not stable"
