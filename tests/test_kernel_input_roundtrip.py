"""Tests for kernel input export and round-trip verification."""
from __future__ import annotations

import hashlib
import json
import tempfile
import os
import numpy as np
import pytest

from certificates.make_certificate import (
    export_kernel_input,
    load_kernel_input,
    _encode_matrix_base64,
    _decode_matrix_base64,
    _hash_matrix_bytes,
)

# Mark all tests in this module as unit tests (Tier 1)
pytestmark = pytest.mark.unit


def test_encode_decode_matrix_round_trip() -> None:
    """Test that encoding and decoding a matrix preserves values."""
    np.random.seed(42)
    M = np.random.randn(5, 7)
    
    # Encode
    encoded = _encode_matrix_base64(M)
    
    # Decode
    M_decoded = _decode_matrix_base64(encoded, 5, 7)
    
    # Should be exactly equal (bit-for-bit via big-endian float64)
    np.testing.assert_array_equal(M, M_decoded)


def test_matrix_hash_stability() -> None:
    """Test that matrix hash is stable across multiple calls."""
    np.random.seed(123)
    M = np.random.randn(4, 6)
    
    # Compute hash multiple times
    hashes = [_hash_matrix_bytes(M) for _ in range(5)]
    
    # All hashes should be identical
    assert len(set(hashes)) == 1, "Matrix hashes are not stable"
    
    # Hash should be valid SHA256 (64 hex characters)
    assert len(hashes[0]) == 64
    assert all(c in '0123456789abcdef' for c in hashes[0])


def test_export_load_kernel_input_round_trip() -> None:
    """Test that exporting and loading kernel input preserves data and SHA256."""
    np.random.seed(456)
    X_aug = np.random.randn(10, 8)
    trace_id = "test_trace_123"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "kernel_input.json")
        
        # Export
        exported = export_kernel_input(
            X_aug=X_aug,
            trace_id=trace_id,
            output_path=output_path,
            embedder_id="test_embedder",
            rank=5,
        )
        
        # Load
        X_aug_loaded, metadata = load_kernel_input(output_path)
        
        # Verify matrix data matches
        np.testing.assert_array_almost_equal(X_aug, X_aug_loaded, decimal=10)
        
        # Verify metadata
        assert metadata["trace_id"] == trace_id
        assert metadata["parameters"]["rank"] == 5
        assert metadata["metadata"]["embedder_id"] == "test_embedder"
        
        # Verify SHA256 matches
        original_hash = exported["observables"]["X_aug"]["sha256"]
        computed_hash = _hash_matrix_bytes(X_aug_loaded)
        assert original_hash == computed_hash, "SHA256 hash mismatch after round-trip"


def test_export_with_precomputed_operator() -> None:
    """Test exporting kernel input with pre-computed operator."""
    np.random.seed(789)
    X_aug = np.random.randn(8, 6)
    A_precompute = np.random.randn(5, 5)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "kernel_input.json")
        
        # Export with operator
        exported = export_kernel_input(
            X_aug=X_aug,
            trace_id="test_with_operator",
            output_path=output_path,
            A_precompute=A_precompute,
        )
        
        # Verify operator is included
        assert exported["koopman_fit"] is not None
        assert "A_precompute" in exported["koopman_fit"]
        
        # Verify operator SHA256
        A_hash = exported["koopman_fit"]["A_precompute"]["sha256"]
        computed_hash = _hash_matrix_bytes(A_precompute)
        assert A_hash == computed_hash


def test_export_with_external_subspace() -> None:
    """Test exporting kernel input with external subspace."""
    np.random.seed(101)
    X_aug = np.random.randn(7, 5)
    external_subspace = np.random.randn(7, 3)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "kernel_input.json")
        
        # Export with subspace
        exported = export_kernel_input(
            X_aug=X_aug,
            trace_id="test_with_subspace",
            output_path=output_path,
            external_subspace=external_subspace,
        )
        
        # Verify subspace is included
        assert exported["external_subspace"] is not None
        assert exported["external_subspace"]["rows"] == 7
        assert exported["external_subspace"]["cols"] == 3
        
        # Verify subspace SHA256
        subspace_hash = exported["external_subspace"]["sha256"]
        computed_hash = _hash_matrix_bytes(external_subspace)
        assert subspace_hash == computed_hash


def test_kernel_input_integrity_check_detects_corruption() -> None:
    """Test that load_kernel_input detects data corruption via SHA256."""
    np.random.seed(202)
    X_aug = np.random.randn(6, 4)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "kernel_input.json")
        
        # Export
        export_kernel_input(
            X_aug=X_aug,
            trace_id="test_corruption",
            output_path=output_path,
        )
        
        # Corrupt the file
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        # Change the SHA256 to a wrong value
        data["observables"]["X_aug"]["sha256"] = "0" * 64
        
        with open(output_path, 'w') as f:
            json.dump(data, f)
        
        # Load should raise ValueError due to integrity check failure
        with pytest.raises(ValueError, match="Integrity check failed"):
            load_kernel_input(output_path)


def test_matrix_hash_deterministic_across_copies() -> None:
    """Test that matrix hash is the same for different array objects with same values."""
    np.random.seed(303)
    M1 = np.random.randn(5, 5)
    M2 = M1.copy()
    M3 = np.array(M1)
    
    hash1 = _hash_matrix_bytes(M1)
    hash2 = _hash_matrix_bytes(M2)
    hash3 = _hash_matrix_bytes(M3)
    
    assert hash1 == hash2 == hash3, "Hash should be deterministic for identical values"


def test_encode_matrix_base64_endianness() -> None:
    """Test that encoding uses big-endian format as specified."""
    M = np.array([[1.0, 2.0], [3.0, 4.0]])
    encoded = _encode_matrix_base64(M)
    
    # Decode manually to verify endianness
    import base64
    raw_bytes = base64.b64decode(encoded)
    # Should decode as big-endian float64
    decoded = np.frombuffer(raw_bytes, dtype='>f8')
    expected = M.flatten()
    
    np.testing.assert_array_almost_equal(decoded, expected, decimal=10)


def test_kernel_input_json_schema_version() -> None:
    """Test that exported kernel input includes schema version."""
    np.random.seed(404)
    X_aug = np.random.randn(5, 5)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "kernel_input.json")
        
        exported = export_kernel_input(
            X_aug=X_aug,
            trace_id="test_schema",
            output_path=output_path,
        )
        
        assert "schema_version" in exported
        assert exported["schema_version"] == "1.0"
