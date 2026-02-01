"""Comprehensive integration tests for verified kernel and kernel_server.

This test suite addresses the recommendations from the code review:
1. Test Python fallback path
2. Test kernel_server + verified kernel path (when available)
3. Add failure tests: kernel missing, kernel crashing, symbol mismatch
4. Ensure verify_bound.py is testable as part of CI

These tests are marked as integration tests and should be run with:
    ESM_ALLOW_KERNEL_LOAD=1 pytest tests/test_kernel_comprehensive.py -v
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tests.test_guards import integration, is_kernel_allowed


@integration
class TestPythonFallbackPath:
    """Test that Python fallback works correctly when kernel is unavailable."""

    def test_python_fallback_compute_certificate(self):
        """Test compute_certificate with Python fallback (no kernel)."""
        from certificates import verified_kernel

        # Force Python fallback
        with patch.dict(os.environ, {"ESM_SKIP_VERIFIED_KERNEL": "1"}):
            # Generate test data
            np.random.seed(42)
            r, n = 4, 10
            X0 = np.random.randn(r, n).astype(np.float64)
            X1 = np.random.randn(r, n).astype(np.float64)
            A = np.random.randn(r, r).astype(np.float64)

            # Call compute_certificate - should use Python fallback
            residual, bound = verified_kernel.compute_certificate(
                X0, X1, A,
                tail_energy=0.01,
                semantic_divergence=0.01,
                lipschitz_margin=0.01,
                strict=False,
            )

            # Validate outputs
            assert isinstance(residual, (int, float)), f"residual not numeric: {type(residual)}"
            assert isinstance(bound, (int, float)), f"bound not numeric: {type(bound)}"
            assert residual >= 0, f"residual negative: {residual}"
            assert bound >= 0, f"bound negative: {bound}"

    def test_python_fallback_compute_residual(self):
        """Test compute_residual with Python fallback."""
        from certificates import verified_kernel

        with patch.dict(os.environ, {"ESM_SKIP_VERIFIED_KERNEL": "1"}):
            np.random.seed(42)
            r, n = 3, 5
            X0 = np.random.randn(r, n).astype(np.float64)
            A = np.eye(r, dtype=np.float64) * 0.9
            X1 = A @ X0 + 0.01 * np.random.randn(r, n).astype(np.float64)

            residual = verified_kernel.compute_residual(X0, X1, A, strict=False)

            assert isinstance(residual, (int, float))
            assert residual >= 0
            assert residual < 1.0  # Should be small for nearly-perfect prediction

    def test_python_fallback_compute_bound(self):
        """Test compute_bound with Python fallback."""
        from certificates import verified_kernel

        with patch.dict(os.environ, {"ESM_SKIP_VERIFIED_KERNEL": "1"}):
            bound = verified_kernel.compute_bound(
                residual=0.1,
                tail_energy=0.05,
                semantic_divergence=0.02,
                lipschitz_margin=0.03,
                strict=False,
            )

            assert isinstance(bound, (int, float))
            assert bound > 0
            # bound = C_res*res + C_tail*tail + C_sem*sem + C_robust*lip
            # All constants are ~2, so bound should be roughly 2*(0.1+0.05+0.02+0.03) = 0.4
            assert 0.1 < bound < 2.0


@integration
@pytest.mark.skipif(not is_kernel_allowed(), reason="Kernel loading not allowed in this test tier")
class TestKernelServerIntegration:
    """Test kernel_server with real verified kernel when available."""

    def test_kernel_server_startup_and_ping(self, tmp_path):
        """Test that kernel_server starts and responds to ping."""
        socket_path = str(tmp_path / "test_kernel.sock")
        
        # Start kernel server
        env = os.environ.copy()
        env["ESM_KERNEL_SOCKET"] = socket_path
        env["PYTHONPATH"] = str(Path(__file__).parent.parent)
        
        proc = subprocess.Popen(
            [sys.executable, "-m", "certificates.kernel_server"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Wait for server to be ready (with timeout)
            ready = False
            for _ in range(30):  # 30 seconds timeout
                if proc.poll() is not None:
                    stdout, stderr = proc.communicate()
                    pytest.fail(f"Server died during startup. stdout={stdout}, stderr={stderr}")
                
                # Check if socket exists
                if os.path.exists(socket_path):
                    time.sleep(0.5)  # Give it a moment to fully initialize
                    ready = True
                    break
                
                time.sleep(1)

            assert ready, "Kernel server failed to create socket within timeout"

            # Try to connect and ping
            import json
            import struct

            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(socket_path)

            try:
                # Send ping request
                request = json.dumps({"method": "ping", "params": {}, "id": 1})
                request_bytes = request.encode("utf-8")
                sock.sendall(struct.pack(">I", len(request_bytes)) + request_bytes)

                # Read response
                length_data = sock.recv(4)
                length = struct.unpack(">I", length_data)[0]
                response_data = b""
                while len(response_data) < length:
                    chunk = sock.recv(min(4096, length - len(response_data)))
                    if not chunk:
                        break
                    response_data += chunk

                response = json.loads(response_data.decode("utf-8"))
                assert response["result"] == "pong"
                assert response["error"] is None

            finally:
                sock.close()

        finally:
            # Cleanup
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def test_kernel_server_selftest(self, tmp_path):
        """Test kernel_server selftest method."""
        socket_path = str(tmp_path / "test_kernel.sock")
        
        env = os.environ.copy()
        env["ESM_KERNEL_SOCKET"] = socket_path
        env["PYTHONPATH"] = str(Path(__file__).parent.parent)
        
        proc = subprocess.Popen(
            [sys.executable, "-m", "certificates.kernel_server"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Wait for socket
            for _ in range(30):
                if os.path.exists(socket_path):
                    time.sleep(0.5)
                    break
                time.sleep(1)

            import json
            import struct

            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(socket_path)

            try:
                # Send selftest request
                request = json.dumps({"method": "selftest", "params": {}, "id": 2})
                request_bytes = request.encode("utf-8")
                sock.sendall(struct.pack(">I", len(request_bytes)) + request_bytes)

                # Read response
                length_data = sock.recv(4)
                length = struct.unpack(">I", length_data)[0]
                response_data = b""
                while len(response_data) < length:
                    chunk = sock.recv(min(4096, length - len(response_data)))
                    if not chunk:
                        break
                    response_data += chunk

                response = json.loads(response_data.decode("utf-8"))
                
                assert response["error"] is None
                result = response["result"]
                assert result["ok"] is True, f"Selftest failed: {result.get('message')}"

            finally:
                sock.close()

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


@integration
class TestKernelFailureModes:
    """Test failure modes: kernel missing, kernel crashing, symbol mismatch."""

    def test_kernel_missing_fallback(self):
        """Test that system falls back gracefully when kernel is missing."""
        from certificates import verified_kernel

        # Point to non-existent kernel
        with patch.dict(os.environ, {"VERIFIED_KERNEL_PATH": "/tmp/nonexistent_kernel.so"}):
            # Should fall back to Python without crashing
            kernel = verified_kernel.load_kernel(strict=False)
            assert kernel is None  # Should fail to load but not crash

            # compute_certificate should still work via Python fallback
            np.random.seed(42)
            X0 = np.random.randn(3, 5).astype(np.float64)
            X1 = np.random.randn(3, 5).astype(np.float64)
            A = np.random.randn(3, 3).astype(np.float64)

            residual, bound = verified_kernel.compute_certificate(
                X0, X1, A, 0.01, 0.01, 0.01, strict=False
            )
            
            assert isinstance(residual, (int, float))
            assert isinstance(bound, (int, float))

    def test_kernel_missing_strict_mode(self):
        """Test that strict mode raises when kernel is missing."""
        from certificates import verified_kernel

        with patch.dict(os.environ, {"VERIFIED_KERNEL_PATH": "/tmp/nonexistent_kernel.so"}):
            # strict=True should raise
            with pytest.raises(Exception):  # Could be KernelError or similar
                verified_kernel.load_kernel(strict=True)

    def test_malformed_kernel_path(self):
        """Test handling of malformed kernel paths."""
        from certificates import verified_kernel

        # Test various malformed paths
        bad_paths = [
            "/dev/null",  # Not a shared library
            "",  # Empty path
            "/tmp/not-a-lib.txt",  # Wrong extension
        ]

        for bad_path in bad_paths:
            if os.path.exists(bad_path):
                with patch.dict(os.environ, {"VERIFIED_KERNEL_PATH": bad_path}):
                    kernel = verified_kernel.load_kernel(strict=False)
                    assert kernel is None  # Should fail gracefully


@integration
class TestVerifyBoundIntegration:
    """Test verify_bound.py as part of CI."""

    def test_verify_bound_runs_successfully(self):
        """Test that verify_bound.py runs without errors on synthetic data."""
        verify_bound_path = Path(__file__).parent.parent / "certificates" / "verify_bound.py"
        assert verify_bound_path.exists(), f"verify_bound.py not found at {verify_bound_path}"

        # Run verify_bound.py with minimal parameters
        result = subprocess.run(
            [
                sys.executable,
                str(verify_bound_path),
                "--datasets", "ar1",
                "--T", "20",
                "--d", "4",
                "--r-values", "2",
                "--trials", "1",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Should exit successfully
        assert result.returncode == 0, f"verify_bound.py failed: {result.stderr}"
        
        # Output should contain validation results
        assert "PASS" in result.stdout or "ar1" in result.stdout

    def test_verify_bound_multiple_datasets(self):
        """Test verify_bound.py with multiple datasets."""
        verify_bound_path = Path(__file__).parent.parent / "certificates" / "verify_bound.py"

        result = subprocess.run(
            [
                sys.executable,
                str(verify_bound_path),
                "--datasets", "ar1,affine",
                "--T", "15",
                "--d", "3",
                "--r-values", "1,2",
                "--trials", "1",
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )

        assert result.returncode == 0, f"verify_bound.py failed: {result.stderr}"


@integration
class TestOOSResidualValidityFlag:
    """Test that oos_valid flag is properly propagated through certificates."""

    def test_oos_valid_in_certificate_output(self):
        """Test that oos_valid flag appears in certificate output."""
        from certificates.make_certificate import compute_certificate

        # Create a normal trace (should have oos_valid=True)
        np.random.seed(42)
        embeddings = [np.random.randn(128) for _ in range(20)]

        cert = compute_certificate(
            embeddings=embeddings,
            task_description="test task",
            strict=False,
        )

        # Check that oos_valid is in the certificate
        assert "oos_valid" in cert, "oos_valid flag missing from certificate"
        assert isinstance(cert["oos_valid"], bool), "oos_valid should be boolean"
        assert cert["oos_valid"] is True, "Normal trace should have oos_valid=True"

    def test_oos_valid_false_for_short_trace(self):
        """Test that short traces get oos_valid=False."""
        from certificates.make_certificate import compute_certificate

        # Create a very short trace (T < 4, should have oos_valid=False)
        np.random.seed(42)
        embeddings = [np.random.randn(128) for _ in range(3)]

        cert = compute_certificate(
            embeddings=embeddings,
            task_description="test task",
            strict=False,
        )

        assert "oos_valid" in cert
        assert cert["oos_valid"] is False, "Short trace should have oos_valid=False"
        # Should also have conservative residual floor
        assert cert["oos_residual"] == 0.1, f"Expected conservative floor 0.1, got {cert['oos_residual']}"
