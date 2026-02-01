"""Tests for the KernelPort adapter module.

These tests verify:
1. Adapter factory returns a valid adapter
2. Python fallback works when verified kernel is unavailable
3. Self-test validates adapter correctness
4. Thread-safety of adapter initialization
"""
from __future__ import annotations

import threading
from unittest import mock

import numpy as np
import pytest


class TestKernelAdapter:
    """Test suite for kernel adapter functionality."""

    def test_make_kernel_adapter_returns_port(self):
        """Factory should return a KernelAdapterBase instance."""
        from esmassessor.kernel_adapter import (
            KernelAdapterBase,
            make_kernel_adapter,
            reset_kernel_adapter,
        )

        reset_kernel_adapter()
        adapter = make_kernel_adapter(prefer_verified=False)
        assert isinstance(adapter, KernelAdapterBase)

    def test_python_adapter_selftest(self):
        """Python fallback adapter should pass self-test."""
        from esmassessor.kernel_adapter import PythonKernelAdapter

        adapter = PythonKernelAdapter()
        assert adapter.selftest() is True
        assert adapter.is_verified is False

    def test_python_adapter_compute_residual(self):
        """Python adapter should compute valid residual."""
        from esmassessor.kernel_adapter import PythonKernelAdapter

        adapter = PythonKernelAdapter()

        # Create test data
        np.random.seed(123)
        r, n = 4, 3
        X0 = np.random.randn(r, n).astype(np.float64)
        A = np.random.randn(r, r).astype(np.float64)
        X1 = A @ X0 + 0.01 * np.random.randn(r, n).astype(np.float64)

        res = adapter.compute_residual(X0, X1, A)
        assert isinstance(res, float)
        assert res >= 0

    def test_python_adapter_compute_bound(self):
        """Python adapter should compute valid bound."""
        from esmassessor.kernel_adapter import PythonKernelAdapter

        adapter = PythonKernelAdapter()

        bound = adapter.compute_bound(
            residual=0.01,
            tail_energy=0.02,
            semantic_divergence=0.03,
            lipschitz_margin=0.04,
        )
        assert isinstance(bound, float)
        assert bound >= 0

    def test_python_adapter_compute_certificate(self):
        """Python adapter should compute valid certificate tuple."""
        from esmassessor.kernel_adapter import PythonKernelAdapter

        adapter = PythonKernelAdapter()

        np.random.seed(456)
        r, n = 3, 2
        X0 = np.random.randn(r, n).astype(np.float64)
        A = np.random.randn(r, r).astype(np.float64)
        X1 = A @ X0

        res, bound = adapter.compute_certificate(X0, X1, A, 0.01, 0.01, 0.01)
        assert isinstance(res, float)
        assert isinstance(bound, float)
        assert res >= 0
        assert bound >= 0

    def test_adapter_caching(self):
        """Factory should cache and reuse adapter instance."""
        from esmassessor.kernel_adapter import (
            get_kernel_adapter,
            make_kernel_adapter,
            reset_kernel_adapter,
        )

        reset_kernel_adapter()
        adapter1 = make_kernel_adapter(prefer_verified=False)
        adapter2 = get_kernel_adapter()
        adapter3 = make_kernel_adapter(prefer_verified=False)

        assert adapter1 is adapter2
        assert adapter2 is adapter3

    def test_reset_clears_adapter(self):
        """reset_kernel_adapter should clear cached adapter."""
        from esmassessor.kernel_adapter import (
            make_kernel_adapter,
            reset_kernel_adapter,
        )

        reset_kernel_adapter()
        adapter1 = make_kernel_adapter(prefer_verified=False)

        reset_kernel_adapter()
        adapter2 = make_kernel_adapter(prefer_verified=False)

        # After reset, should get a new instance
        assert adapter1 is not adapter2

    def test_thread_safety(self):
        """Concurrent calls to make_kernel_adapter should be safe."""
        from esmassessor.kernel_adapter import (
            KernelAdapterBase,
            make_kernel_adapter,
            reset_kernel_adapter,
        )

        reset_kernel_adapter()
        results = []
        errors = []

        def get_adapter():
            try:
                adapter = make_kernel_adapter(prefer_verified=False)
                results.append(adapter)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_adapter) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10

        # All threads should get the same instance
        first = results[0]
        for r in results[1:]:
            assert r is first


class TestKernelAdapterError:
    """Test error handling in kernel adapter."""

    def test_adapter_error_is_runtime_error(self):
        """KernelAdapterError should inherit from RuntimeError."""
        from esmassessor.kernel_adapter import KernelAdapterError

        err = KernelAdapterError("test error")
        assert isinstance(err, RuntimeError)
        assert str(err) == "test error"
