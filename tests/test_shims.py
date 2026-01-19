"""Tests for transformers compatibility shims.

These tests verify:
1. Shims are applied correctly
2. Shims are idempotent (safe to apply multiple times)
3. DynamicCache methods work after shimming
"""
from __future__ import annotations

import pytest


class TestShims:
    """Test suite for transformers compatibility shims."""

    def test_apply_shims_returns_dict(self):
        """apply_transformers_shims should return a status dict."""
        from tools.real_agents_hf.shims import apply_transformers_shims, reset_shims_state

        reset_shims_state()
        result = apply_transformers_shims()

        assert isinstance(result, dict)

    def test_apply_shims_idempotent(self):
        """Applying shims multiple times should be safe."""
        from tools.real_agents_hf.shims import apply_transformers_shims, reset_shims_state

        reset_shims_state()

        result1 = apply_transformers_shims()
        result2 = apply_transformers_shims()

        # Second call should indicate already applied
        assert result2.get("already_applied") is True

    def test_dynamic_cache_seen_tokens_shim(self):
        """DynamicCache.seen_tokens should be available after shimming."""
        from tools.real_agents_hf.shims import apply_transformers_shims

        apply_transformers_shims()

        try:
            from transformers.cache_utils import DynamicCache

            assert hasattr(DynamicCache, "seen_tokens")
        except ImportError:
            pytest.skip("transformers not installed")

    def test_dynamic_cache_get_max_length_shim(self):
        """DynamicCache.get_max_length should be available after shimming."""
        from tools.real_agents_hf.shims import apply_transformers_shims

        apply_transformers_shims()

        try:
            from transformers.cache_utils import DynamicCache

            assert hasattr(DynamicCache, "get_max_length")
        except ImportError:
            pytest.skip("transformers not installed")

    def test_dynamic_cache_get_usable_length_shim(self):
        """DynamicCache.get_usable_length should be available after shimming."""
        from tools.real_agents_hf.shims import apply_transformers_shims

        apply_transformers_shims()

        try:
            from transformers.cache_utils import DynamicCache

            assert hasattr(DynamicCache, "get_usable_length")
        except ImportError:
            pytest.skip("transformers not installed")

    def test_get_usable_length_accepts_positional_arg(self):
        """get_usable_length should accept positional seq_length argument."""
        from tools.real_agents_hf.shims import apply_transformers_shims

        apply_transformers_shims()

        try:
            from transformers.cache_utils import DynamicCache

            cache = DynamicCache()

            # Should not raise when called with positional arg
            result = cache.get_usable_length(10)
            assert isinstance(result, int)
            assert result >= 0
        except ImportError:
            pytest.skip("transformers not installed")

    def test_get_usable_length_accepts_keyword_arg(self):
        """get_usable_length should accept keyword seq_length argument."""
        from tools.real_agents_hf.shims import apply_transformers_shims

        apply_transformers_shims()

        try:
            from transformers.cache_utils import DynamicCache

            cache = DynamicCache()

            # Should not raise when called with keyword arg
            result = cache.get_usable_length(seq_length=10)
            assert isinstance(result, int)
            assert result >= 0
        except ImportError:
            pytest.skip("transformers not installed")

    def test_get_usable_length_no_args(self):
        """get_usable_length should work with no arguments."""
        from tools.real_agents_hf.shims import apply_transformers_shims

        apply_transformers_shims()

        try:
            from transformers.cache_utils import DynamicCache

            cache = DynamicCache()

            # Should not raise when called with no args
            result = cache.get_usable_length()
            assert isinstance(result, int)
            assert result >= 0
        except ImportError:
            pytest.skip("transformers not installed")


class TestShimsEdgeCases:
    """Edge case tests for shims."""

    def test_reset_allows_reapplication(self):
        """reset_shims_state should allow shims to be reapplied."""
        from tools.real_agents_hf.shims import apply_transformers_shims, reset_shims_state

        reset_shims_state()
        result1 = apply_transformers_shims()

        reset_shims_state()
        result2 = apply_transformers_shims()

        # After reset, should not show already_applied
        assert result2.get("already_applied") is not True
