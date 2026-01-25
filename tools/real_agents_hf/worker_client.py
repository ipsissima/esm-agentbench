#!/usr/bin/env python3
"""Client for communicating with the persistent model worker.

This module provides functions to:
- Check if a worker is running
- Send generation requests with timeout
- Start/restart the worker process

Usage in agent_loop:
    from .worker_client import generate_via_worker, is_worker_available

    if is_worker_available():
        response = generate_via_worker(prompt, max_tokens=128, timeout=300)
    else:
        # Fall back to direct generation
        response = backend.generate(prompt)
"""
from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Configuration
SOCKET_PATH = os.environ.get("ESM_WORKER_SOCKET", "/tmp/esm_model_worker.sock")
USE_PERSISTENT_WORKER = os.environ.get("ESM_GEN_USE_PERSISTENT", "0") == "1"

# Worker process tracking
_worker_process: Optional[subprocess.Popen] = None
_worker_pid_file: Optional[str] = None


class WorkerError(Exception):
    """Raised when worker returns an error."""
    pass


class WorkerTimeout(Exception):
    """Raised when worker doesn't respond in time."""
    pass


class WorkerUnavailable(Exception):
    """Raised when worker is not running."""
    pass


def is_worker_available() -> bool:
    """Check if a persistent worker is running and responsive."""
    if not USE_PERSISTENT_WORKER:
        return False

    if not os.path.exists(SOCKET_PATH):
        return False

    # Try a quick connection test
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect(SOCKET_PATH)
        sock.close()
        return True
    except (socket.error, socket.timeout):
        return False


def start_worker(
    model_id: str = "microsoft/Phi-3-mini-128k-instruct",
    revision: Optional[str] = None,
    socket_path: Optional[str] = None,
) -> subprocess.Popen:
    """Start a new worker process.

    Returns the Popen object so caller can track/kill it.
    """
    global _worker_process

    worker_script = Path(__file__).parent / "model_worker.py"

    env = os.environ.copy()
    env["ESM_MODEL_ID"] = model_id
    if revision:
        env["ESM_MODEL_REVISION"] = revision
    if socket_path:
        env["ESM_WORKER_SOCKET"] = socket_path

    logger.info(f"Starting worker: {model_id}")

    _worker_process = subprocess.Popen(
        [sys.executable, str(worker_script)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for socket to appear (model loading)
    socket_to_check = socket_path or SOCKET_PATH
    max_wait = 120  # 2 minutes for model loading
    start = time.time()

    while time.time() - start < max_wait:
        if os.path.exists(socket_to_check):
            # Socket exists, try to connect
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(2.0)
                sock.connect(socket_to_check)
                sock.close()
                logger.info(f"Worker ready after {time.time() - start:.1f}s")
                return _worker_process
            except (socket.error, socket.timeout):
                pass

        # Check if process died
        if _worker_process.poll() is not None:
            stdout, stderr = _worker_process.communicate()
            raise RuntimeError(
                f"Worker process died during startup. "
                f"Exit code: {_worker_process.returncode}\n"
                f"stderr: {stderr.decode()[-1000:]}"
            )

        time.sleep(1)

    raise RuntimeError(f"Worker did not become ready within {max_wait}s")


def stop_worker() -> None:
    """Stop the current worker process."""
    global _worker_process

    if _worker_process is None:
        return

    logger.info("Stopping worker...")

    # Try graceful termination first
    _worker_process.terminate()
    try:
        _worker_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        logger.warning("Worker didn't terminate, killing...")
        _worker_process.kill()
        _worker_process.wait(timeout=5)

    _worker_process = None

    # Clean up socket
    if os.path.exists(SOCKET_PATH):
        try:
            os.unlink(SOCKET_PATH)
        except Exception:
            pass


def restart_worker(
    model_id: str = "microsoft/Phi-3-mini-128k-instruct",
    revision: Optional[str] = None,
) -> subprocess.Popen:
    """Stop existing worker and start a new one."""
    stop_worker()
    return start_worker(model_id, revision)


def generate_via_worker(
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.7,
    stop_sequences: Optional[List[str]] = None,
    timeout: int = 300,
    socket_path: Optional[str] = None,
) -> str:
    """Send generation request to worker and wait for response.

    Parameters
    ----------
    prompt : str
        The prompt to generate from
    max_tokens : int
        Maximum tokens to generate
    temperature : float
        Generation temperature
    stop_sequences : list of str, optional
        Stop sequences
    timeout : int
        Timeout in seconds
    socket_path : str, optional
        Override socket path

    Returns
    -------
    str
        Generated text

    Raises
    ------
    WorkerTimeout
        If worker doesn't respond within timeout
    WorkerError
        If worker returns an error
    WorkerUnavailable
        If worker is not running
    """
    socket_to_use = socket_path or SOCKET_PATH

    if not os.path.exists(socket_to_use):
        raise WorkerUnavailable(f"Worker socket not found: {socket_to_use}")

    # Build request
    request = {
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop_sequences:
        request["stop_sequences"] = stop_sequences

    request_data = (json.dumps(request) + "\n").encode()

    # Connect and send
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)

    try:
        sock.connect(socket_to_use)
        sock.sendall(request_data)

        # Receive response
        response_data = b""
        start = time.time()

        while time.time() - start < timeout:
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response_data += chunk
                if b"\n" in chunk:
                    break
            except socket.timeout:
                continue

        if not response_data:
            raise WorkerTimeout(f"No response from worker within {timeout}s")

        # Parse response
        response = json.loads(response_data.decode().strip())

        if not response.get("ok"):
            error_msg = response.get("error", "Unknown error")
            raise WorkerError(f"Worker error: {error_msg}")

        return response["output"]

    except socket.timeout:
        raise WorkerTimeout(f"Worker timed out after {timeout}s")
    except socket.error as e:
        raise WorkerUnavailable(f"Socket error: {e}")
    finally:
        sock.close()


def generate_with_worker_fallback(
    backend,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.7,
    stop_sequences: Optional[List[str]] = None,
    timeout: int = 300,
) -> str:
    """Try worker first, fall back to direct generation.

    This is the main entry point for agent_loop to use.
    """
    if USE_PERSISTENT_WORKER and is_worker_available():
        try:
            logger.debug("Using persistent worker for generation")
            return generate_via_worker(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
                timeout=timeout,
            )
        except (WorkerTimeout, WorkerError, WorkerUnavailable) as e:
            logger.warning(f"Worker failed ({e}), falling back to direct generation")

    # Fall back to direct generation
    logger.debug("Using direct generation (no worker)")
    return backend.generate(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop_sequences,
    )
