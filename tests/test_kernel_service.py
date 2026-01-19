"""Tests for kernel-as-service architecture.

These tests verify that the kernel service properly isolates native code
and handles failures gracefully.

Running:
    # Unit tests (mocked service)
    pytest tests/test_kernel_service.py -v -m unit

    # Integration tests (real service)
    ESM_KERNEL_SERVICE=1 pytest tests/test_kernel_service.py -v -m integration
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.test_guards import (
    unit,
    integration,
    kernel,
    skip_if_no_kernel,
    is_kernel_allowed,
)


@unit
class TestKernelServiceProtocol:
    """Test kernel service protocol and encoding."""

    def test_array_encoding_roundtrip(self):
        """Test that arrays survive encoding/decoding."""
        from certificates.kernel_service import _encode_array, _decode_array

        original = np.random.randn(4, 3).astype(np.float64)
        encoded = _encode_array(original)
        decoded = _decode_array(encoded)

        np.testing.assert_array_equal(original, decoded)

    def test_array_encoding_preserves_dtype(self):
        """Test that encoding preserves array dtype."""
        from certificates.kernel_service import _encode_array, _decode_array

        original = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
        decoded = _decode_array(_encode_array(original))

        assert decoded.dtype == original.dtype

    def test_kernel_request_serialization(self):
        """Test KernelRequest JSON serialization."""
        from certificates.kernel_service import KernelRequest

        request = KernelRequest(
            method="compute_residual",
            params={"X0": "encoded_array", "X1": "encoded_array", "A": "encoded_array"},
            id=42,
        )

        json_str = request.to_json()
        assert '"method": "compute_residual"' in json_str
        assert '"id": 42' in json_str

    def test_kernel_response_deserialization(self):
        """Test KernelResponse JSON deserialization."""
        from certificates.kernel_service import KernelResponse

        json_str = '{"result": 0.123, "error": null, "id": 42}'
        response = KernelResponse.from_json(json_str)

        assert response.result == 0.123
        assert response.error is None
        assert response.id == 42

    def test_kernel_response_with_error(self):
        """Test KernelResponse with error."""
        from certificates.kernel_service import KernelResponse

        json_str = '{"result": null, "error": "Test error", "id": 1}'
        response = KernelResponse.from_json(json_str)

        assert response.result is None
        assert response.error == "Test error"


@unit
class TestKernelServiceClient:
    """Test kernel service client without actual server."""

    def test_client_initialization(self):
        """Test that client can be created."""
        from certificates.kernel_service import KernelServiceClient

        # Don't auto-start server
        client = KernelServiceClient(auto_start=False)

        assert client.socket_path is not None
        assert client.timeout > 0

    def test_is_service_mode_detection(self):
        """Test service mode detection."""
        from certificates.kernel_service import is_service_mode

        # Default is off
        original = os.environ.get("ESM_KERNEL_SERVICE")
        try:
            os.environ["ESM_KERNEL_SERVICE"] = "0"
            assert not is_service_mode()

            os.environ["ESM_KERNEL_SERVICE"] = "1"
            assert is_service_mode()
        finally:
            if original is None:
                os.environ.pop("ESM_KERNEL_SERVICE", None)
            else:
                os.environ["ESM_KERNEL_SERVICE"] = original

    def test_client_not_running_detection(self):
        """Test that client can detect server not running."""
        from certificates.kernel_service import KernelServiceClient

        # Use a socket path that definitely doesn't exist
        client = KernelServiceClient(
            socket_path="/tmp/nonexistent_kernel_socket_12345.sock",
            auto_start=False,
        )

        assert not client._is_server_running()


@integration
@kernel
class TestKernelServiceIntegration:
    """Integration tests for kernel service (requires real server)."""

    @skip_if_no_kernel
    @pytest.mark.timeout(30)
    def test_server_startup_and_ping(self):
        """Test that server can start and respond to ping."""
        from certificates.kernel_service import (
            KernelServiceClient,
            reset_kernel_client,
        )

        reset_kernel_client()

        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = Path(tmpdir) / "kernel.sock"

            try:
                client = KernelServiceClient(
                    socket_path=str(socket_path),
                    auto_start=True,
                )

                # Should be able to ping
                response = client._send_request("ping", {})
                assert response.result == "pong"

            finally:
                if 'client' in locals():
                    client.shutdown()

    @skip_if_no_kernel
    @pytest.mark.timeout(30)
    def test_server_selftest(self):
        """Test that server passes selftest."""
        from certificates.kernel_service import (
            KernelServiceClient,
            reset_kernel_client,
        )

        reset_kernel_client()

        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = Path(tmpdir) / "kernel.sock"

            try:
                client = KernelServiceClient(
                    socket_path=str(socket_path),
                    auto_start=True,
                )

                ok, message = client.selftest()
                assert ok, f"Selftest failed: {message}"

            finally:
                if 'client' in locals():
                    client.shutdown()

    @skip_if_no_kernel
    @pytest.mark.timeout(30)
    def test_compute_certificate_via_service(self):
        """Test computing certificate via service."""
        from certificates.kernel_service import (
            KernelServiceClient,
            reset_kernel_client,
        )

        reset_kernel_client()

        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = Path(tmpdir) / "kernel.sock"

            try:
                client = KernelServiceClient(
                    socket_path=str(socket_path),
                    auto_start=True,
                )

                # Generate test data
                np.random.seed(42)
                X0 = np.random.randn(4, 2).astype(np.float64)
                A = np.random.randn(4, 4).astype(np.float64)
                X1 = A @ X0 + 0.001 * np.random.randn(4, 2).astype(np.float64)

                # Compute via service
                residual, bound = client.compute_certificate(
                    X0, X1, A, 0.01, 0.01, 0.01
                )

                # Verify results
                assert residual >= 0
                assert bound >= 0
                assert np.isfinite(residual)
                assert np.isfinite(bound)

            finally:
                if 'client' in locals():
                    client.shutdown()

    @skip_if_no_kernel
    @pytest.mark.timeout(30)
    def test_server_isolation_on_bad_input(self):
        """Test that server handles bad input gracefully."""
        from certificates.kernel_service import (
            KernelServiceClient,
            KernelServiceError,
            reset_kernel_client,
        )

        reset_kernel_client()

        with tempfile.TemporaryDirectory() as tmpdir:
            socket_path = Path(tmpdir) / "kernel.sock"

            try:
                client = KernelServiceClient(
                    socket_path=str(socket_path),
                    auto_start=True,
                )

                # Send invalid request
                response = client._send_request("invalid_method", {})
                assert response.error is not None
                assert "Unknown method" in response.error

                # Server should still be responsive
                response = client._send_request("ping", {})
                assert response.result == "pong"

            finally:
                if 'client' in locals():
                    client.shutdown()


@unit
class TestKernelServiceFallback:
    """Test fallback behavior when service is not available."""

    def test_can_use_direct_kernel_when_service_unavailable(self):
        """Test that we can fall back to direct kernel usage."""
        from esmassessor.kernel_adapter import (
            PythonKernelAdapter,
            make_kernel_adapter,
            reset_kernel_adapter,
        )

        reset_kernel_adapter()

        # Should get Python adapter when kernel not available
        adapter = make_kernel_adapter(prefer_verified=False)
        assert isinstance(adapter, PythonKernelAdapter)

        # Should still work
        np.random.seed(42)
        X0 = np.random.randn(4, 2).astype(np.float64)
        A = np.random.randn(4, 4).astype(np.float64)
        X1 = A @ X0 + 0.001 * np.random.randn(4, 2).astype(np.float64)

        res, bound = adapter.compute_certificate(X0, X1, A, 0.01, 0.01, 0.01)

        assert res >= 0
        assert bound >= 0

        reset_kernel_adapter()
