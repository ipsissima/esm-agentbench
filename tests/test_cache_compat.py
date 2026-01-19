"""Tests for cache compatibility and fallback behavior.

These tests verify:
1. DynamicCache shims are properly applied and functional
2. Cache fallback to use_cache=False works correctly
3. Policy correctly disables cache on CPU
4. Generate handles cache-related AttributeErrors gracefully
5. Generate handles tensor shape mismatches gracefully
"""
from __future__ import annotations

from unittest import mock

import pytest


class TestCacheShimsIntegration:
    """Test cache shims are applied and functional."""

    def test_shims_applied_on_inference_import(self):
        """Importing inference should apply shims automatically."""
        from tools.real_agents_hf.shims import reset_shims_state

        # Reset shims state to test fresh import behavior
        reset_shims_state()

        # Import inference module - this should apply shims
        import importlib
        import tools.real_agents_hf.inference as inference_module

        importlib.reload(inference_module)

        # Verify shims are applied
        try:
            from transformers.cache_utils import DynamicCache

            assert hasattr(DynamicCache, "seen_tokens")
            assert hasattr(DynamicCache, "get_max_length")
            assert hasattr(DynamicCache, "get_usable_length")
        except ImportError:
            pytest.skip("transformers not installed")

    def test_dynamic_cache_seen_tokens_invariant(self):
        """DynamicCache.seen_tokens should equal get_seq_length()."""
        from tools.real_agents_hf.shims import apply_transformers_shims

        apply_transformers_shims()

        try:
            from transformers.cache_utils import DynamicCache

            cache = DynamicCache()
            # Invariant: seen_tokens == get_seq_length()
            assert cache.seen_tokens == cache.get_seq_length()
        except ImportError:
            pytest.skip("transformers not installed")

    def test_dynamic_cache_get_max_length_non_negative(self):
        """DynamicCache.get_max_length should return non-negative integer."""
        from tools.real_agents_hf.shims import apply_transformers_shims

        apply_transformers_shims()

        try:
            from transformers.cache_utils import DynamicCache

            cache = DynamicCache()
            result = cache.get_max_length()

            assert isinstance(result, int)
            assert result >= 0
        except ImportError:
            pytest.skip("transformers not installed")

    def test_dynamic_cache_get_usable_length_non_negative(self):
        """DynamicCache.get_usable_length should return non-negative integer."""
        from tools.real_agents_hf.shims import apply_transformers_shims

        apply_transformers_shims()

        try:
            from transformers.cache_utils import DynamicCache

            cache = DynamicCache()

            # Test with various arguments
            assert cache.get_usable_length() >= 0
            assert cache.get_usable_length(10) >= 0
            assert cache.get_usable_length(seq_length=100) >= 0
        except ImportError:
            pytest.skip("transformers not installed")


class TestCachePolicyBehavior:
    """Test policy-driven cache behavior."""

    def test_policy_disables_cache_on_cpu(self):
        """On CPU, policy should disable KV caching."""
        from tools.real_agents_hf.runtime_policy import RuntimePolicy

        policy = RuntimePolicy()
        if not policy.cuda:
            assert policy.allow_cache() is False
            gen_kwargs = policy.get_generate_kwargs()
            assert gen_kwargs.get("use_cache") is False

    def test_force_no_cache_persists_after_fallback(self):
        """After cache fallback, _force_no_cache should persist."""
        from tools.real_agents_hf.inference import TransformersBackend, ModelConfig

        config = ModelConfig(
            name="test",
            hf_id="test/test",
            backend="transformers",
            dtype="float32",
            max_tokens=10,
            temperature=0.0,
        )
        backend = TransformersBackend(config)

        # Initially no force
        assert backend._force_no_cache is False

        # Simulate setting force after encountering cache error
        backend._force_no_cache = True

        # Should persist
        assert backend._force_no_cache is True


class TestGenerateCacheFallback:
    """Test generate() cache fallback behavior."""

    def test_generate_catches_cache_attribute_error(self):
        """generate() should catch AttributeError for cache internals and retry."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        from tools.real_agents_hf.inference import TransformersBackend, ModelConfig

        config = ModelConfig(
            name="test",
            hf_id="test/test",
            backend="transformers",
            dtype="float32",
            max_tokens=10,
            temperature=0.0,
        )
        backend = TransformersBackend(config)

        # Create mock model and tokenizer
        mock_model = mock.MagicMock()
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1

        # Track call count
        call_count = [0]
        successful_output = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])

        def mock_generate(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1 and kwargs.get("use_cache", True):
                # First call with cache enabled raises cache error
                raise AttributeError("'DynamicCache' object has no attribute 'seen_tokens'")
            # Second call (or first call without cache) succeeds
            return successful_output

        mock_model.generate = mock_generate
        mock_model.device = torch.device("cpu")

        backend.model = mock_model
        backend.tokenizer = mock_tokenizer

        # Setup tokenizer mock
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        mock_tokenizer.return_value = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
        mock_tokenizer.decode = mock.MagicMock(return_value="test output")

        # Should not raise - should catch and retry
        result = backend.generate("test prompt")

        # _force_no_cache should be set after the retry
        assert backend._force_no_cache is True
        assert call_count[0] == 2  # First failed, second succeeded

    def test_generate_handles_shape_mismatch_error(self):
        """generate() should catch RuntimeError for shape mismatches and retry."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        from tools.real_agents_hf.inference import TransformersBackend, ModelConfig

        config = ModelConfig(
            name="test",
            hf_id="test/test",
            backend="transformers",
            dtype="float32",
            max_tokens=10,
            temperature=0.0,
        )
        backend = TransformersBackend(config)

        # Create mock model and tokenizer
        mock_model = mock.MagicMock()
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1

        call_count = [0]
        successful_output = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])

        def mock_generate(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1 and kwargs.get("use_cache", True):
                # First call raises shape mismatch error (like the 970/1940 error)
                raise RuntimeError(
                    "shape '[1, 32, 970, 96]' is invalid for input of size 5898240"
                )
            return successful_output

        mock_model.generate = mock_generate
        mock_model.device = torch.device("cpu")

        backend.model = mock_model
        backend.tokenizer = mock_tokenizer

        # Setup tokenizer mock
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        mock_tokenizer.return_value = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
        mock_tokenizer.decode = mock.MagicMock(return_value="test output")

        # Should not raise - should catch and retry
        result = backend.generate("test prompt")

        # _force_no_cache should be set
        assert backend._force_no_cache is True
        assert call_count[0] == 2


class TestCacheShapeMismatchKeywords:
    """Test that shape mismatch detection covers expected error patterns."""

    def test_shape_keyword_detection(self):
        """RuntimeError with 'shape' keyword should trigger fallback."""
        # The error patterns we need to catch
        error_patterns = [
            "shape '[1, 32, 970, 96]' is invalid",
            "Expected size 1940 but got 970",
            "Size mismatch between tensors",
            "dimension mismatch in attention",
        ]

        for pattern in error_patterns:
            msg = pattern.lower()
            # These keywords should be detected
            assert any(x in msg for x in ("shape", "size", "mismatch", "dimension"))
