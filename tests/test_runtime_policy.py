"""Tests for RuntimePolicy module.

These tests verify:
1. Policy correctly detects environment capabilities
2. Policy methods return expected values for CPU/GPU environments
3. Policy helpers generate correct model kwargs
"""
from __future__ import annotations

from unittest import mock

import pytest


class TestRuntimePolicy:
    """Test suite for RuntimePolicy functionality."""

    def test_policy_initialization(self):
        """RuntimePolicy should initialize without errors."""
        from tools.real_agents_hf.runtime_policy import RuntimePolicy

        policy = RuntimePolicy()
        assert hasattr(policy, "cuda")
        assert hasattr(policy, "have_bnb")
        assert hasattr(policy, "have_flash")
        assert hasattr(policy, "have_accelerate")

    def test_policy_str_repr(self):
        """Policy should have readable string representation."""
        from tools.real_agents_hf.runtime_policy import RuntimePolicy

        policy = RuntimePolicy()
        s = str(policy)
        assert "RuntimePolicy" in s
        assert "cuda=" in s

    def test_allow_quant_requires_cuda_and_bnb(self):
        """Quantization should only be allowed with CUDA and bitsandbytes."""
        from tools.real_agents_hf.runtime_policy import RuntimePolicy

        policy = RuntimePolicy()
        # If either is missing, should return False
        if not policy.cuda or not policy.have_bnb:
            assert policy.allow_quant() is False
        else:
            assert policy.allow_quant() is True

    def test_attn_impl_returns_valid_value(self):
        """attn_impl should return a valid attention implementation."""
        from tools.real_agents_hf.runtime_policy import RuntimePolicy

        policy = RuntimePolicy()
        attn = policy.attn_impl()
        assert attn in ("flash_attention_2", "eager", "sdpa")

    def test_attn_impl_eager_on_cpu(self):
        """On CPU, attn_impl should return 'eager'."""
        from tools.real_agents_hf.runtime_policy import RuntimePolicy

        policy = RuntimePolicy()
        if not policy.cuda:
            assert policy.attn_impl() == "eager"

    def test_allow_cache_false_on_cpu(self):
        """On CPU, caching should be disabled."""
        from tools.real_agents_hf.runtime_policy import RuntimePolicy

        policy = RuntimePolicy()
        if not policy.cuda:
            assert policy.allow_cache() is False

    def test_recommended_dtype_float32_on_cpu(self):
        """On CPU, recommended dtype should be float32."""
        from tools.real_agents_hf.runtime_policy import RuntimePolicy

        policy = RuntimePolicy()
        if not policy.cuda:
            assert policy.recommended_dtype("float16") == "float32"
            assert policy.recommended_dtype("bfloat16") == "float32"
            assert policy.recommended_dtype("float32") == "float32"

    def test_recommended_dtype_preserves_on_gpu(self):
        """On GPU, recommended dtype should preserve requested dtype."""
        from tools.real_agents_hf.runtime_policy import RuntimePolicy

        policy = RuntimePolicy()
        if policy.cuda:
            assert policy.recommended_dtype("float16") == "float16"
            assert policy.recommended_dtype("bfloat16") == "bfloat16"
            assert policy.recommended_dtype("float32") == "float32"

    def test_get_model_kwargs_structure(self):
        """get_model_kwargs should return valid kwargs dict."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        from tools.real_agents_hf.runtime_policy import RuntimePolicy

        policy = RuntimePolicy()
        kwargs = policy.get_model_kwargs()

        assert isinstance(kwargs, dict)
        assert "trust_remote_code" in kwargs
        assert kwargs["trust_remote_code"] is True
        assert "torch_dtype" in kwargs
        assert "attn_implementation" in kwargs

    def test_get_model_kwargs_no_quant_on_cpu(self):
        """On CPU, model kwargs should not include quantization."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        from tools.real_agents_hf.runtime_policy import RuntimePolicy

        policy = RuntimePolicy()
        if not policy.cuda:
            kwargs = policy.get_model_kwargs(load_in_4bit=True, load_in_8bit=True)
            assert "quantization_config" not in kwargs
            assert kwargs.get("load_in_8bit") is not True

    def test_get_generate_kwargs_no_cache_on_cpu(self):
        """On CPU, generate kwargs should disable cache."""
        from tools.real_agents_hf.runtime_policy import RuntimePolicy

        policy = RuntimePolicy()
        if not policy.cuda:
            kwargs = policy.get_generate_kwargs()
            assert kwargs.get("use_cache") is False

    def test_global_policy_singleton(self):
        """get_runtime_policy should return same instance."""
        from tools.real_agents_hf.runtime_policy import (
            get_runtime_policy,
            reset_runtime_policy,
        )

        reset_runtime_policy()
        policy1 = get_runtime_policy()
        policy2 = get_runtime_policy()

        assert policy1 is policy2

    def test_reset_clears_singleton(self):
        """reset_runtime_policy should clear cached policy."""
        from tools.real_agents_hf.runtime_policy import (
            get_runtime_policy,
            reset_runtime_policy,
        )

        reset_runtime_policy()
        policy1 = get_runtime_policy()

        reset_runtime_policy()
        policy2 = get_runtime_policy()

        assert policy1 is not policy2


class TestRuntimePolicyMocked:
    """Tests with mocked environment."""

    def test_policy_with_mocked_cuda(self):
        """Test policy behavior with mocked CUDA availability."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        from tools.real_agents_hf.runtime_policy import RuntimePolicy

        with mock.patch("torch.cuda.is_available", return_value=True):
            policy = RuntimePolicy()
            # Note: have_bnb and have_flash depend on actual imports
            # so we can only check cuda flag
            # Policy was constructed with cuda=True

    def test_allow_device_map_requires_cuda_and_accelerate(self):
        """device_map should only be allowed with CUDA and accelerate."""
        from tools.real_agents_hf.runtime_policy import RuntimePolicy

        policy = RuntimePolicy()
        result = policy.allow_device_map()

        if policy.cuda and policy.have_accelerate:
            assert result is True
        else:
            assert result is False
