#!/usr/bin/env python3
"""Kernel Service Server - Runs verified kernel in isolated process.

This script runs as a subprocess and hosts the verified kernel behind
a Unix socket RPC interface. If the kernel crashes, only this process
dies - the parent process remains alive and can detect the failure.

Security Features:
    - Socket file permissions (0600 by default)
    - Optional HMAC-based authentication for requests
    - Request rate limiting to prevent DoS
    - Maximum request size enforcement

Usage:
    python kernel_server.py

Environment:
    ESM_KERNEL_SOCKET         Unix socket path (default: /tmp/esm_kernel.sock)
    ESM_KERNEL_SOCKET_PERMS   Socket permissions octal (default: 0600)
    ESM_KERNEL_AUTH_TOKEN     Optional HMAC authentication token
    ESM_KERNEL_MAX_REQUESTS   Max requests per client per second (default: 100)
    VERIFIED_KERNEL_PATH      Path to kernel shared library
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import logging
import os
import signal
import socket
import stat
import struct
import sys
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.DEBUG,
    format="[kernel-server] %(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SOCKET_PATH = "/tmp/esm_kernel.sock"
DEFAULT_SOCKET_PERMS = 0o600  # Owner read/write only
DEFAULT_MAX_REQUESTS_PER_SEC = 100


def _decode_array(data: str) -> np.ndarray:
    """Decode numpy array from base64 string."""
    buf = io.BytesIO(base64.b64decode(data))
    return np.load(buf, allow_pickle=False)


def _encode_array(arr: np.ndarray) -> str:
    """Encode numpy array to base64 string for JSON transport."""
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class KernelServer:
    """RPC server for verified kernel operations with authentication and rate limiting."""

    def __init__(
        self,
        socket_path: str,
        socket_perms: int = DEFAULT_SOCKET_PERMS,
        auth_token: Optional[str] = None,
        max_requests_per_sec: int = DEFAULT_MAX_REQUESTS_PER_SEC,
    ):
        self.socket_path = socket_path
        self.socket_perms = socket_perms
        self.auth_token = auth_token
        self.max_requests_per_sec = max_requests_per_sec
        self._running = False
        self._kernel = None
        self._vk = None  # verified_kernel module
        self._server_socket = None
        
        # Rate limiting: track requests per client
        self._client_requests: Dict[Any, list] = defaultdict(list)
        self._rate_limit_lock = threading.Lock()

    def _load_kernel(self) -> bool:
        """Load the verified kernel library."""
        try:
            from certificates import verified_kernel as vk
            self._vk = vk
            self._kernel = vk.load_kernel(strict=False)
            if self._kernel is not None:
                logger.info("Verified kernel loaded successfully")
                return True
            else:
                logger.warning("Verified kernel not available, using Python fallback")
                return True  # Fallback is OK
        except Exception as exc:
            logger.error(f"Failed to load kernel: {exc}")
            return False

    def _check_rate_limit(self, client_addr: Any) -> bool:
        """Check if client has exceeded rate limit.
        
        Returns:
            True if client is within rate limit, False if exceeded.
        """
        with self._rate_limit_lock:
            now = time.time()
            # Clean old requests (older than 1 second)
            self._client_requests[client_addr] = [
                t for t in self._client_requests[client_addr]
                if now - t < 1.0
            ]
            
            # Check rate
            if len(self._client_requests[client_addr]) >= self.max_requests_per_sec:
                return False
            
            # Record this request
            self._client_requests[client_addr].append(now)
            return True

    def _verify_auth(self, request: dict) -> bool:
        """Verify HMAC authentication if enabled.
        
        Returns:
            True if auth passes or is disabled, False if auth fails.
        """
        if self.auth_token is None:
            # No authentication required
            return True
        
        # Extract HMAC from request
        provided_hmac = request.get("hmac")
        if not provided_hmac:
            logger.warning("Request missing HMAC authentication")
            return False
        
        # Compute expected HMAC
        # HMAC is over method + params (excluding the hmac field itself)
        auth_data = {
            "method": request.get("method", ""),
            "params": request.get("params", {}),
            "id": request.get("id", 0),
        }
        message = json.dumps(auth_data, sort_keys=True).encode("utf-8")
        expected_hmac = hmac.new(
            self.auth_token.encode("utf-8"),
            message,
            hashlib.sha256
        ).hexdigest()
        
        # Constant-time comparison
        if not hmac.compare_digest(provided_hmac, expected_hmac):
            logger.warning("HMAC authentication failed")
            return False
        
        return True

    def _handle_request(self, data: bytes, client_addr: Any) -> bytes:
        """Handle a single RPC request and return response."""
        try:
            request = json.loads(data.decode("utf-8"))
            method = request.get("method", "")
            params = request.get("params", {})
            request_id = request.get("id", 0)

            logger.debug(f"Handling request: {method}")

            # Verify authentication if enabled
            if not self._verify_auth(request):
                response = {
                    "error": "Authentication failed",
                    "result": None,
                    "id": request_id,
                }
                return json.dumps(response).encode("utf-8")

            # Dispatch to method handler
            if method == "ping":
                result = "pong"
            elif method == "shutdown":
                self._running = False
                result = {"ok": True}
            elif method == "selftest":
                result = self._handle_selftest()
            elif method == "compute_residual":
                result = self._handle_compute_residual(params)
            elif method == "compute_bound":
                result = self._handle_compute_bound(params)
            elif method == "compute_certificate":
                result = self._handle_compute_certificate(params)
            else:
                response = {
                    "error": f"Unknown method: {method}",
                    "result": None,
                    "id": request_id,
                }
                return json.dumps(response).encode("utf-8")

            response = {"result": result, "error": None, "id": request_id}
            return json.dumps(response).encode("utf-8")

        except Exception as exc:
            logger.error(f"Request handling error: {exc}\n{traceback.format_exc()}")
            response = {
                "error": str(exc),
                "result": None,
                "id": request.get("id", 0) if isinstance(request, dict) else 0,
            }
            return json.dumps(response).encode("utf-8")

    def _handle_selftest(self) -> Dict[str, Any]:
        """Run kernel self-test."""
        try:
            # Generate test data
            np.random.seed(42)
            r, n = 4, 2
            X0 = np.random.randn(r, n).astype(np.float64)
            A = np.random.randn(r, r).astype(np.float64)
            X1 = A @ X0 + 0.001 * np.random.randn(r, n).astype(np.float64)

            # Test compute_certificate
            res, bound = self._vk.compute_certificate(
                X0, X1, A, 0.01, 0.01, 0.01, strict=False
            )

            # Validate results
            if not isinstance(res, (int, float)):
                return {"ok": False, "message": f"residual not numeric: {type(res)}"}
            if not isinstance(bound, (int, float)):
                return {"ok": False, "message": f"bound not numeric: {type(bound)}"}
            if res < 0:
                return {"ok": False, "message": f"residual negative: {res}"}
            if bound < 0:
                return {"ok": False, "message": f"bound negative: {bound}"}

            return {
                "ok": True,
                "message": f"Self-test passed: res={res:.6f}, bound={bound:.6f}",
                "verified": self._kernel is not None,
            }

        except Exception as exc:
            return {"ok": False, "message": f"Self-test exception: {exc}"}

    def _handle_compute_residual(self, params: Dict[str, Any]) -> float:
        """Handle compute_residual request."""
        X0 = _decode_array(params["X0"])
        X1 = _decode_array(params["X1"])
        A = _decode_array(params["A"])

        # Call verified kernel (or Python fallback)
        return self._vk.compute_residual(X0, X1, A, strict=False)

    def _handle_compute_bound(self, params: Dict[str, Any]) -> float:
        """Handle compute_bound request."""
        return self._vk.compute_bound(
            residual=params["residual"],
            tail_energy=params["tail_energy"],
            semantic_divergence=params["semantic_divergence"],
            lipschitz_margin=params["lipschitz_margin"],
            strict=False,
        )

    def _handle_compute_certificate(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Handle compute_certificate request."""
        X0 = _decode_array(params["X0"])
        X1 = _decode_array(params["X1"])
        A = _decode_array(params["A"])

        res, bound = self._vk.compute_certificate(
            X0,
            X1,
            A,
            params["tail_energy"],
            params["semantic_divergence"],
            params["lipschitz_margin"],
            strict=False,
        )

        return {"residual": float(res), "bound": float(bound)}

    def _handle_client(self, conn: socket.socket, addr: Any) -> None:
        """Handle a single client connection."""
        try:
            while self._running:
                # Check rate limit
                if not self._check_rate_limit(addr):
                    logger.warning(f"Rate limit exceeded for client {addr}")
                    error_response = json.dumps({
                        "error": "Rate limit exceeded",
                        "result": None,
                        "id": 0,
                    }).encode("utf-8")
                    conn.sendall(struct.pack(">I", len(error_response)) + error_response)
                    break

                # Read request length (4-byte big-endian)
                length_data = conn.recv(4)
                if len(length_data) < 4:
                    break

                length = struct.unpack(">I", length_data)[0]
                if length > 10 * 1024 * 1024:  # 10MB limit
                    logger.error(f"Request too large: {length}")
                    break

                # Read request data
                data = b""
                while len(data) < length:
                    chunk = conn.recv(min(4096, length - len(data)))
                    if not chunk:
                        break
                    data += chunk

                if len(data) < length:
                    break

                # Handle request
                response = self._handle_request(data, addr)

                # Send response (length-prefixed)
                conn.sendall(struct.pack(">I", len(response)) + response)

        except Exception as exc:
            logger.error(f"Client handler error: {exc}")
        finally:
            conn.close()

    def run(self) -> None:
        """Run the kernel server."""
        # Load kernel first
        if not self._load_kernel():
            logger.error("Failed to initialize kernel, exiting")
            sys.exit(1)

        # Clean up old socket
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

        # Create server socket
        self._server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind(self.socket_path)
        
        # Set socket permissions for security
        try:
            os.chmod(self.socket_path, self.socket_perms)
            logger.info(f"Socket permissions set to {oct(self.socket_perms)}")
        except OSError as e:
            logger.warning(f"Failed to set socket permissions: {e}")
        
        self._server_socket.listen(5)
        self._server_socket.settimeout(1.0)  # Allow periodic shutdown check

        self._running = True
        
        auth_status = "enabled" if self.auth_token else "disabled"
        logger.info(
            f"Kernel server listening on {self.socket_path} "
            f"(auth: {auth_status}, rate_limit: {self.max_requests_per_sec}/s)"
        )

        # Signal readiness
        print("KERNEL_SERVER_READY", flush=True)

        while self._running:
            try:
                conn, addr = self._server_socket.accept()
                # Handle client in separate thread
                thread = threading.Thread(
                    target=self._handle_client, args=(conn, addr), daemon=True
                )
                thread.start()
            except socket.timeout:
                continue
            except Exception as exc:
                if self._running:
                    logger.error(f"Accept error: {exc}")

        # Cleanup
        self._server_socket.close()
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

        logger.info("Kernel server shutdown complete")


def main():
    """Main entry point."""
    socket_path = os.environ.get("ESM_KERNEL_SOCKET", DEFAULT_SOCKET_PATH)
    
    # Parse socket permissions from environment (octal string like "0600")
    socket_perms_str = os.environ.get("ESM_KERNEL_SOCKET_PERMS", "0600")
    try:
        socket_perms = int(socket_perms_str, 8)  # Parse as octal
    except ValueError:
        logger.warning(f"Invalid socket permissions '{socket_perms_str}', using default 0600")
        socket_perms = DEFAULT_SOCKET_PERMS
    
    # Optional authentication token
    auth_token = os.environ.get("ESM_KERNEL_AUTH_TOKEN")
    if auth_token:
        logger.info("HMAC authentication enabled")
    
    # Rate limiting
    max_requests_str = os.environ.get("ESM_KERNEL_MAX_REQUESTS", str(DEFAULT_MAX_REQUESTS_PER_SEC))
    try:
        max_requests_per_sec = int(max_requests_str)
    except ValueError:
        logger.warning(f"Invalid max requests '{max_requests_str}', using default {DEFAULT_MAX_REQUESTS_PER_SEC}")
        max_requests_per_sec = DEFAULT_MAX_REQUESTS_PER_SEC

    # Handle signals
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    server = KernelServer(
        socket_path=socket_path,
        socket_perms=socket_perms,
        auth_token=auth_token,
        max_requests_per_sec=max_requests_per_sec,
    )
    server.run()


if __name__ == "__main__":
    main()
