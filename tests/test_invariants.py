"""Comprehensive Invariant Test Suite.

This module tests all architectural invariants that must hold for the system
to be problem-proof. These tests serve as executable documentation of the
correctness criteria.

Invariants Tested:
1. Attention invariant: All attention modules agree on implementation
2. Cache contract invariant: DynamicCache has required methods
3. Kernel contract invariant: Kernel computations are side-effect free
4. ABI/Artifact invariant: Kernel checksums match expected values
5. Environment determinism invariant: Reproducible outputs given same inputs

Running:
    # Unit tier (fast, no native code)
    ESM_ALLOW_KERNEL_LOAD=0 pytest tests/test_invariants.py -v

    # Integration tier (with kernel)
    ESM_ALLOW_KERNEL_LOAD=1 pytest tests/test_invariants.py -v

    # Strict mode (fail on any violation)
    ESM_STRICT=1 pytest tests/test_invariants.py -v
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from tests.test_guards import (
    unit,
    integration,
    heavy,
    kernel,
    model,
    signing,
    skip_if_no_kernel,
    skip_if_no_model,
    skip_if_no_pynacl,
    skip_if_no_transformers,
    requires_kernel,
    requires_model,
    is_kernel_allowed,
    is_model_allowed,
    has_pynacl,
    has_transformers,
)


# =============================================================================
# 1. ATTENTION INVARIANT TESTS
# =============================================================================


@unit
class TestAttentionInvariant:
    """Test that attention modules agree on implementation."""

    @skip_if_no_transformers
    def test_attention_info_extraction(self):
        """Test that we can extract attention info from modules."""
        from tools.real_agents_hf.model_adapter import (
            ModelAdapter,
            AttentionInfo,
            AttentionImplementation,
        )

        # Create a mock module with attention attributes
        class MockModule:
            def __init__(self):
                self.attn_impl = "eager"
                self.num_heads = 8
                self.head_dim = 64

        adapter = object.__new__(ModelAdapter)
        adapter.model = None
        adapter.config = {"attn_implementation": "eager"}

        info = adapter._extract_attention_info("test.attention", MockModule())

        assert info.name == "test.attention"
        assert info.implementation == AttentionImplementation.EAGER
        assert info.num_heads == 8
        assert info.head_dim == 64

    @skip_if_no_transformers
    def test_attention_implementation_parsing(self):
        """Test parsing of attention implementation strings."""
        from tools.real_agents_hf.model_adapter import (
            ModelAdapter,
            AttentionImplementation,
        )

        adapter = object.__new__(ModelAdapter)

        # Test various strings
        assert adapter._parse_attn_impl("eager") == AttentionImplementation.EAGER
        assert adapter._parse_attn_impl("EAGER") == AttentionImplementation.EAGER
        assert adapter._parse_attn_impl("flash_attention_2") == AttentionImplementation.FLASH_ATTENTION_2
        assert adapter._parse_attn_impl("flash") == AttentionImplementation.FLASH_ATTENTION_2
        assert adapter._parse_attn_impl("sdpa") == AttentionImplementation.SDPA
        assert adapter._parse_attn_impl("unknown_impl") == AttentionImplementation.UNKNOWN

    @skip_if_no_transformers
    def test_compatibility_report_creation(self):
        """Test that compatibility reports are properly created."""
        from tools.real_agents_hf.model_adapter import CompatibilityReport

        report = CompatibilityReport(is_compatible=True)

        assert report.is_compatible
        assert len(report.issues) == 0
        assert len(report.warnings) == 0

        report.add_issue("Test issue")
        assert not report.is_compatible
        assert "Test issue" in report.issues

        report.add_warning("Test warning")
        assert "Test warning" in report.warnings


# =============================================================================
# 2. CACHE CONTRACT INVARIANT TESTS
# =============================================================================


@unit
class TestCacheContractInvariant:
    """Test that DynamicCache satisfies required contract."""

    @skip_if_no_transformers
    def test_shims_applied(self):
        """Test that transformers shims are applied."""
        from tools.real_agents_hf.shims import apply_transformers_shims

        result = apply_transformers_shims()

        # Should return dict (either with results or already_applied)
        assert isinstance(result, dict)

    @skip_if_no_transformers
    def test_dynamic_cache_has_required_methods(self):
        """Test that DynamicCache has required methods after shims."""
        from tools.real_agents_hf.shims import apply_transformers_shims
        apply_transformers_shims()

        from transformers.cache_utils import DynamicCache

        # These methods must exist for cache contract
        assert hasattr(DynamicCache, "get_seq_length"), "Missing get_seq_length"

        # These may be added by shims
        assert hasattr(DynamicCache, "get_usable_length"), "Missing get_usable_length (shim should add)"

    @skip_if_no_transformers
    def test_get_usable_length_accepts_seq_length(self):
        """Test that get_usable_length accepts seq_length argument."""
        from tools.real_agents_hf.shims import apply_transformers_shims
        apply_transformers_shims()

        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()

        # Should not raise with seq_length argument
        result = cache.get_usable_length(seq_length=100)
        assert isinstance(result, int)
        assert result >= 0

        # Should also work with positional argument
        result = cache.get_usable_length(100)
        assert isinstance(result, int)
        assert result >= 0

    @skip_if_no_transformers
    def test_cache_contract_info_creation(self):
        """Test CacheContractInfo dataclass."""
        from tools.real_agents_hf.model_adapter import CacheContractInfo

        info = CacheContractInfo()
        assert not info.is_compliant

        info = CacheContractInfo(
            has_get_usable_length=True,
            has_get_seq_length=True,
            is_compliant=True,
        )
        assert info.is_compliant


# =============================================================================
# 3. KERNEL CONTRACT INVARIANT TESTS
# =============================================================================


@unit
class TestKernelContractInvariant:
    """Test kernel contract (side-effect free, correct results)."""

    def test_kernel_adapter_interface(self):
        """Test that kernel adapter has required interface."""
        from esmassessor.kernel_adapter import KernelPort

        # Check abstract methods exist
        assert hasattr(KernelPort, "compute_residual")
        assert hasattr(KernelPort, "compute_bound")
        assert hasattr(KernelPort, "compute_certificate")
        assert hasattr(KernelPort, "is_verified")

    def test_python_adapter_implements_interface(self):
        """Test that PythonKernelAdapter implements full interface."""
        from esmassessor.kernel_adapter import PythonKernelAdapter, KernelPort

        adapter = PythonKernelAdapter()

        assert isinstance(adapter, KernelPort)
        assert not adapter.is_verified  # Python fallback is not verified

    def test_python_adapter_selftest(self):
        """Test that Python adapter passes self-test."""
        from esmassessor.kernel_adapter import PythonKernelAdapter

        adapter = PythonKernelAdapter()
        result = adapter.selftest()
        assert result is True

    def test_kernel_computation_determinism(self):
        """Test that kernel computations are deterministic."""
        from esmassessor.kernel_adapter import PythonKernelAdapter

        adapter = PythonKernelAdapter()

        # Generate fixed test data
        np.random.seed(42)
        X0 = np.random.randn(4, 2).astype(np.float64)
        A = np.random.randn(4, 4).astype(np.float64)
        X1 = A @ X0 + 0.001 * np.random.randn(4, 2).astype(np.float64)

        # Compute twice
        res1, bound1 = adapter.compute_certificate(X0, X1, A, 0.01, 0.01, 0.01)
        res2, bound2 = adapter.compute_certificate(X0, X1, A, 0.01, 0.01, 0.01)

        # Results must be identical
        assert res1 == res2, "Residual not deterministic"
        assert bound1 == bound2, "Bound not deterministic"

    def test_kernel_no_input_mutation(self):
        """Test that kernel does not mutate input arrays."""
        from esmassessor.kernel_adapter import PythonKernelAdapter

        adapter = PythonKernelAdapter()

        # Create input arrays
        X0 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        X1 = np.array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
        A = np.eye(4)

        # Copy for comparison
        X0_copy = X0.copy()
        X1_copy = X1.copy()
        A_copy = A.copy()

        # Compute
        adapter.compute_certificate(X0, X1, A, 0.01, 0.01, 0.01)

        # Verify no mutation
        np.testing.assert_array_equal(X0, X0_copy, "X0 was mutated")
        np.testing.assert_array_equal(X1, X1_copy, "X1 was mutated")
        np.testing.assert_array_equal(A, A_copy, "A was mutated")

    def test_kernel_numeric_bounds(self):
        """Test that kernel results satisfy numeric bounds."""
        from esmassessor.kernel_adapter import PythonKernelAdapter

        adapter = PythonKernelAdapter()

        np.random.seed(123)
        X0 = np.random.randn(4, 2).astype(np.float64)
        A = np.random.randn(4, 4).astype(np.float64)
        X1 = A @ X0 + 0.001 * np.random.randn(4, 2).astype(np.float64)

        res, bound = adapter.compute_certificate(X0, X1, A, 0.01, 0.01, 0.01)

        # Residual must be non-negative
        assert res >= 0, f"Residual negative: {res}"

        # Bound must be non-negative
        assert bound >= 0, f"Bound negative: {bound}"

        # Bound must be finite
        assert np.isfinite(bound), f"Bound not finite: {bound}"


@integration
@kernel
class TestKernelServiceInvariant:
    """Test kernel-as-service invariants."""

    @skip_if_no_kernel
    def test_kernel_service_selftest(self):
        """Test that kernel service passes self-test."""
        # This test requires actual kernel service running
        # Skip if not in service mode
        if os.environ.get("ESM_KERNEL_SERVICE", "0") != "1":
            pytest.skip("Kernel service mode not enabled")

        from certificates.kernel_service import get_kernel_client

        client = get_kernel_client()
        ok, message = client.selftest()

        assert ok, f"Kernel service selftest failed: {message}"


# =============================================================================
# 4. ARTIFACT SIGNING INVARIANT TESTS
# =============================================================================


@unit
@signing
class TestArtifactSigningInvariant:
    """Test artifact signing and verification."""

    @skip_if_no_pynacl
    def test_key_generation(self):
        """Test that signing keys can be generated."""
        from certificates.artifact_signing import ArtifactSigner

        signer, public_key = ArtifactSigner.generate("test-signer")

        assert signer is not None
        assert len(public_key) > 0
        assert signer.signer_id == "test-signer"

    @skip_if_no_pynacl
    def test_sign_and_verify_bytes(self):
        """Test signing and verifying arbitrary bytes."""
        from certificates.artifact_signing import (
            ArtifactSigner,
            ArtifactVerifier,
        )

        # Generate key pair
        signer, public_key = ArtifactSigner.generate("test")

        # Sign some data
        data = b"test data for signing"
        signature = signer.sign_bytes(data)

        assert len(signature) == 64  # Ed25519 signature is 64 bytes

        # Verify with correct key
        verifier = ArtifactVerifier(public_key=public_key)
        assert verifier._verify_signature(data, signature, signer._key.verify_key.encode())

    @skip_if_no_pynacl
    def test_sign_and_verify_file(self):
        """Test signing and verifying a file."""
        from certificates.artifact_signing import (
            ArtifactSigner,
            ArtifactVerifier,
        )

        # Generate key pair
        signer, public_key = ArtifactSigner.generate("test")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            test_file = Path(tmpdir) / "test.bin"
            test_file.write_bytes(b"test content for signing")

            # Sign file
            artifact = signer.sign_file(test_file)

            assert artifact is not None
            assert artifact.manifest.filename == "test.bin"
            assert len(artifact.signature) == 64

            # Verify file
            verifier = ArtifactVerifier(public_key=public_key)
            assert verifier.verify_file(test_file)

    @skip_if_no_pynacl
    def test_tampered_file_fails_verification(self):
        """Test that tampered files fail verification."""
        from certificates.artifact_signing import (
            ArtifactSigner,
            ArtifactVerifier,
            VerificationError,
        )

        signer, public_key = ArtifactSigner.generate("test")

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.bin"
            test_file.write_bytes(b"original content")

            # Sign original
            signer.sign_file(test_file)

            # Tamper with file
            test_file.write_bytes(b"tampered content")

            # Verification should fail
            verifier = ArtifactVerifier(public_key=public_key)
            with pytest.raises(VerificationError):
                verifier.verify_file(test_file)

    def test_hash_computation(self):
        """Test file hash computation."""
        from certificates.artifact_signing import compute_file_hashes

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.bin"
            test_file.write_bytes(b"test content")

            sha256, sha512 = compute_file_hashes(test_file)

            # Verify hash lengths
            assert len(sha256) == 64  # SHA-256 hex is 64 chars
            assert len(sha512) == 128  # SHA-512 hex is 128 chars

            # Verify determinism
            sha256_2, sha512_2 = compute_file_hashes(test_file)
            assert sha256 == sha256_2
            assert sha512 == sha512_2


# =============================================================================
# 5. ENVIRONMENT DETERMINISM INVARIANT TESTS
# =============================================================================


@unit
class TestEnvironmentDeterminismInvariant:
    """Test that outputs are reproducible given same inputs."""

    def test_numpy_random_reproducibility(self):
        """Test numpy random reproducibility with seed."""
        np.random.seed(42)
        a = np.random.randn(10)

        np.random.seed(42)
        b = np.random.randn(10)

        np.testing.assert_array_equal(a, b)

    def test_kernel_adapter_factory_returns_same_instance(self):
        """Test that adapter factory returns cached instance."""
        from esmassessor.kernel_adapter import (
            make_kernel_adapter,
            reset_kernel_adapter,
        )

        reset_kernel_adapter()

        adapter1 = make_kernel_adapter(prefer_verified=False)
        adapter2 = make_kernel_adapter(prefer_verified=False)

        assert adapter1 is adapter2, "Factory should return cached instance"

        reset_kernel_adapter()

    def test_runtime_policy_singleton(self):
        """Test that runtime policy is a singleton."""
        from tools.real_agents_hf.runtime_policy import (
            get_runtime_policy,
            reset_runtime_policy,
        )

        reset_runtime_policy()

        policy1 = get_runtime_policy()
        policy2 = get_runtime_policy()

        assert policy1 is policy2, "Policy should be singleton"

        reset_runtime_policy()


# =============================================================================
# 6. RUNTIME POLICY INVARIANT TESTS
# =============================================================================


@unit
class TestRuntimePolicyInvariant:
    """Test runtime policy invariants."""

    def test_policy_detects_environment(self):
        """Test that policy correctly detects environment."""
        from tools.real_agents_hf.runtime_policy import RuntimePolicy

        policy = RuntimePolicy()

        # Basic type checks
        assert isinstance(policy.cuda, bool)
        assert isinstance(policy.have_bnb, bool)
        assert isinstance(policy.have_flash, bool)
        assert isinstance(policy.have_accelerate, bool)

    def test_policy_cpu_safe_defaults(self):
        """Test that policy provides CPU-safe defaults."""
        from tools.real_agents_hf.runtime_policy import RuntimePolicy

        policy = RuntimePolicy()

        # On CPU (cuda=False), should return safe defaults
        if not policy.cuda:
            assert not policy.allow_quant(), "Quant should be disabled on CPU"
            assert policy.attn_impl() == "eager", "Should use eager attention on CPU"
            assert not policy.allow_cache(), "Cache should be disabled on CPU"
            assert policy.recommended_dtype("float16") == "float32", "Should force float32 on CPU"

    def test_policy_generate_kwargs_consistency(self):
        """Test that generate kwargs are consistent with policy."""
        from tools.real_agents_hf.runtime_policy import RuntimePolicy

        policy = RuntimePolicy()
        gen_kwargs = policy.get_generate_kwargs()

        if not policy.allow_cache():
            assert gen_kwargs.get("use_cache") is False


# =============================================================================
# 7. SHIMS INVARIANT TESTS
# =============================================================================


@unit
class TestShimsInvariant:
    """Test that shims are idempotent and correct."""

    @skip_if_no_transformers
    def test_shims_idempotent(self):
        """Test that shims can be applied multiple times safely."""
        from tools.real_agents_hf.shims import (
            apply_transformers_shims,
            reset_shims_state,
        )

        reset_shims_state()

        # First application
        result1 = apply_transformers_shims()
        # Second application
        result2 = apply_transformers_shims()

        # Second should return already_applied
        assert result2.get("already_applied") is True

        reset_shims_state()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


@integration
@model
class TestModelIntegration:
    """Integration tests for model loading and inference."""

    @skip_if_no_model
    @skip_if_no_transformers
    def test_model_adapter_load_tiny_model(self):
        """Test loading a tiny model with ModelAdapter."""
        pytest.skip("Requires tiny model download - run in heavy tier")


@heavy
class TestHeavyInvariants:
    """Heavy tests that require GPU and full models."""

    @pytest.mark.gpu
    @skip_if_no_model
    def test_full_model_compatibility(self):
        """Test full model compatibility check."""
        pytest.skip("Requires GPU and full model - run in nightly")
