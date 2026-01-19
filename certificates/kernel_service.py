"""Kernel-as-Service: Process-isolated verified kernel execution.

This module implements process isolation for the formally verified Coq/OCaml kernel.
If the kernel segfaults, only the worker process dies - the main process remains alive.

Architecture:
- KernelServer: Runs in subprocess, loads kernel via ctypes
- KernelClient: Communicates with server via JSON-RPC over Unix socket
- Automatic fallback to in-process mode if service unavailable

Usage:
    # Service mode (default when ESM_KERNEL_SERVICE=1)
    client = KernelServiceClient()
    residual, bound = client.compute_certificate(X0, X1, A, ...)

    # In-process mode (for testing/backwards compat)
    from certificates.verified_kernel import compute_certificate
    residual, bound = compute_certificate(X0, X1, A, ...)

Environment:
    ESM_KERNEL_SERVICE=1  Enable kernel-as-service mode
    ESM_KERNEL_SOCKET     Unix socket path (default: /tmp/esm_kernel.sock)
    ESM_KERNEL_TIMEOUT    RPC timeout in seconds (default: 30)
"""
from __future__ import annotations

import atexit
import base64
import io
import json
import logging
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_SOCKET_PATH = "/tmp/esm_kernel.sock"
DEFAULT_TIMEOUT = 30.0
DEFAULT_STARTUP_TIMEOUT = 10.0


class KernelServiceError(RuntimeError):
    """Raised when kernel service encounters an error."""


@dataclass
class KernelRequest:
    """RPC request to kernel service."""
    method: str
    params: Dict[str, Any]
    id: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class KernelResponse:
    """RPC response from kernel service."""
    result: Optional[Any] = None
    error: Optional[str] = None
    id: int = 0

    @classmethod
    def from_json(cls, data: str) -> "KernelResponse":
        d = json.loads(data)
        return cls(result=d.get("result"), error=d.get("error"), id=d.get("id", 0))


def _encode_array(arr: np.ndarray) -> str:
    """Encode numpy array to base64 string for JSON transport."""
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_array(data: str) -> np.ndarray:
    """Decode numpy array from base64 string."""
    buf = io.BytesIO(base64.b64decode(data))
    return np.load(buf, allow_pickle=False)


class KernelServiceClient:
    """Client for kernel-as-service.

    Communicates with a subprocess that hosts the verified kernel.
    If the kernel crashes, only the subprocess dies.
    """

    def __init__(
        self,
        socket_path: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        auto_start: bool = True,
    ):
        """Initialize kernel service client.

        Parameters
        ----------
        socket_path : str, optional
            Unix socket path for communication.
        timeout : float
            RPC timeout in seconds.
        auto_start : bool
            If True, automatically start server if not running.
        """
        self.socket_path = socket_path or os.environ.get(
            "ESM_KERNEL_SOCKET", DEFAULT_SOCKET_PATH
        )
        self.timeout = timeout
        self._server_process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._lock = threading.Lock()

        if auto_start:
            self._ensure_server_running()

    def _ensure_server_running(self) -> None:
        """Start server if not already running."""
        if self._is_server_running():
            return

        logger.info("Starting kernel service...")
        self._start_server()

    def _is_server_running(self) -> bool:
        """Check if server is responsive."""
        try:
            response = self._send_request("ping", {})
            return response.result == "pong"
        except Exception:
            return False

    def _start_server(self) -> None:
        """Start the kernel server subprocess."""
        # Clean up old socket if exists
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

        # Start server process
        server_script = Path(__file__).parent / "kernel_server.py"
        env = os.environ.copy()
        env["ESM_KERNEL_SOCKET"] = self.socket_path

        self._server_process = subprocess.Popen(
            [sys.executable, str(server_script)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to be ready
        start = time.time()
        while time.time() - start < DEFAULT_STARTUP_TIMEOUT:
            if self._is_server_running():
                logger.info("Kernel service started successfully")
                atexit.register(self.shutdown)
                return
            time.sleep(0.1)

        # Server failed to start
        if self._server_process.poll() is not None:
            _, stderr = self._server_process.communicate()
            raise KernelServiceError(
                f"Kernel server failed to start: {stderr.decode()[:500]}"
            )

        raise KernelServiceError("Kernel server startup timeout")

    def _send_request(self, method: str, params: Dict[str, Any]) -> KernelResponse:
        """Send RPC request to server and wait for response."""
        with self._lock:
            self._request_id += 1
            request = KernelRequest(method=method, params=params, id=self._request_id)

            # Connect to server
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)

            try:
                sock.connect(self.socket_path)

                # Send request (length-prefixed JSON)
                data = request.to_json().encode("utf-8")
                sock.sendall(struct.pack(">I", len(data)) + data)

                # Receive response
                length_data = sock.recv(4)
                if len(length_data) < 4:
                    raise KernelServiceError("Server disconnected")

                length = struct.unpack(">I", length_data)[0]
                response_data = b""
                while len(response_data) < length:
                    chunk = sock.recv(min(4096, length - len(response_data)))
                    if not chunk:
                        raise KernelServiceError("Server disconnected")
                    response_data += chunk

                return KernelResponse.from_json(response_data.decode("utf-8"))

            finally:
                sock.close()

    def selftest(self) -> Tuple[bool, str]:
        """Run kernel self-test.

        Returns
        -------
        Tuple[bool, str]
            (success, diagnostics_message)
        """
        try:
            response = self._send_request("selftest", {})
            if response.error:
                return False, f"Selftest error: {response.error}"
            return response.result.get("ok", False), response.result.get("message", "")
        except Exception as exc:
            return False, f"Selftest failed: {exc}"

    def compute_residual(
        self, X0: np.ndarray, X1: np.ndarray, A: np.ndarray
    ) -> float:
        """Compute residual using verified kernel service.

        Parameters
        ----------
        X0 : np.ndarray
            State matrix at time t.
        X1 : np.ndarray
            State matrix at time t+1.
        A : np.ndarray
            Learned operator.

        Returns
        -------
        float
            The residual error.
        """
        params = {
            "X0": _encode_array(X0.astype(np.float64)),
            "X1": _encode_array(X1.astype(np.float64)),
            "A": _encode_array(A.astype(np.float64)),
        }
        response = self._send_request("compute_residual", params)

        if response.error:
            raise KernelServiceError(f"compute_residual failed: {response.error}")

        return float(response.result)

    def compute_bound(
        self,
        residual: float,
        tail_energy: float,
        semantic_divergence: float,
        lipschitz_margin: float,
    ) -> float:
        """Compute theoretical bound using verified kernel service.

        Parameters
        ----------
        residual : float
            Prediction residual.
        tail_energy : float
            Energy not captured by truncation.
        semantic_divergence : float
            Mean divergence from task embedding.
        lipschitz_margin : float
            Embedding Lipschitz margin.

        Returns
        -------
        float
            The theoretical bound.
        """
        params = {
            "residual": float(residual),
            "tail_energy": float(tail_energy),
            "semantic_divergence": float(semantic_divergence),
            "lipschitz_margin": float(lipschitz_margin),
        }
        response = self._send_request("compute_bound", params)

        if response.error:
            raise KernelServiceError(f"compute_bound failed: {response.error}")

        return float(response.result)

    def compute_certificate(
        self,
        X0: np.ndarray,
        X1: np.ndarray,
        A: np.ndarray,
        tail_energy: float,
        semantic_divergence: float,
        lipschitz_margin: float,
    ) -> Tuple[float, float]:
        """Compute residual and bound using verified kernel service.

        Parameters
        ----------
        X0, X1, A : np.ndarray
            Witness matrices.
        tail_energy, semantic_divergence, lipschitz_margin : float
            Error metrics.

        Returns
        -------
        Tuple[float, float]
            (residual, theoretical_bound)
        """
        params = {
            "X0": _encode_array(X0.astype(np.float64)),
            "X1": _encode_array(X1.astype(np.float64)),
            "A": _encode_array(A.astype(np.float64)),
            "tail_energy": float(tail_energy),
            "semantic_divergence": float(semantic_divergence),
            "lipschitz_margin": float(lipschitz_margin),
        }
        response = self._send_request("compute_certificate", params)

        if response.error:
            raise KernelServiceError(f"compute_certificate failed: {response.error}")

        result = response.result
        return float(result["residual"]), float(result["bound"])

    def shutdown(self) -> None:
        """Shutdown the kernel server."""
        if self._server_process is not None:
            try:
                self._send_request("shutdown", {})
            except Exception:
                pass

            # Give it a moment to exit gracefully
            time.sleep(0.1)

            if self._server_process.poll() is None:
                self._server_process.terminate()
                try:
                    self._server_process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self._server_process.kill()

            self._server_process = None

        # Clean up socket
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

    def __del__(self):
        """Cleanup on garbage collection."""
        try:
            self.shutdown()
        except Exception:
            pass


# Global client instance (lazy initialized)
_kernel_client: Optional[KernelServiceClient] = None
_client_lock = threading.Lock()


def get_kernel_client() -> KernelServiceClient:
    """Get global kernel service client.

    Returns
    -------
    KernelServiceClient
        The singleton kernel client.
    """
    global _kernel_client

    if _kernel_client is not None:
        return _kernel_client

    with _client_lock:
        if _kernel_client is not None:
            return _kernel_client

        _kernel_client = KernelServiceClient()
        return _kernel_client


def is_service_mode() -> bool:
    """Check if kernel-as-service mode is enabled.

    Returns
    -------
    bool
        True if ESM_KERNEL_SERVICE=1
    """
    return os.environ.get("ESM_KERNEL_SERVICE", "0") == "1"


def reset_kernel_client() -> None:
    """Reset the global kernel client (for testing)."""
    global _kernel_client

    with _client_lock:
        if _kernel_client is not None:
            _kernel_client.shutdown()
            _kernel_client = None


__all__ = [
    "KernelServiceClient",
    "KernelServiceError",
    "get_kernel_client",
    "is_service_mode",
    "reset_kernel_client",
]
