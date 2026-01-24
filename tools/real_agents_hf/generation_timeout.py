#!/usr/bin/env python3
"""Generation timeout wrapper for robust model.generate() calls.

This module provides a timeout-protected wrapper around model generation
to prevent indefinite hangs. It uses multiprocessing to isolate the
generate() call and enforce a hard timeout.

IMPORTANT: SIGALRM-based timeouts CANNOT interrupt blocking C library calls
(like PyTorch matrix operations on CPU). For CPU execution, this module uses
a multiprocessing approach that can forcibly terminate hung generation.

Usage:
    from .generation_timeout import generate_with_timeout

    result = generate_with_timeout(
        backend, prompt, max_tokens=100, timeout_seconds=300
    )

Environment variables:
    ESM_GEN_TIMEOUT: Default timeout in seconds (default: 300)
    ESM_GEN_TIMEOUT_ENABLED: Set to "0" to disable timeout wrapper
    ESM_GEN_USE_MP: Set to "1" to force multiprocessing timeout (slower but reliable)
"""
from __future__ import annotations

import logging
import multiprocessing
import os
import signal
import sys
import time
from multiprocessing import Process, Queue
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default timeout: 5 minutes for GPU, 2 minutes for CPU (since tokens are limited)
DEFAULT_TIMEOUT_SECONDS = int(os.getenv("ESM_GEN_TIMEOUT", "300"))
CPU_TIMEOUT_SECONDS = int(os.getenv("ESM_GEN_CPU_TIMEOUT", "120"))

# Allow disabling timeout wrapper via environment
TIMEOUT_ENABLED = os.getenv("ESM_GEN_TIMEOUT_ENABLED", "1") != "0"

# Force multiprocessing-based timeout (more reliable for CPU-bound ops)
FORCE_MP_TIMEOUT = os.getenv("ESM_GEN_USE_MP", "0") == "1"


class GenerationTimeout(Exception):
    """Raised when model.generate() exceeds the timeout."""
    pass


class GenerationError(Exception):
    """Raised when model.generate() fails with an error."""
    pass


def _worker_generate(
    queue: Queue,
    backend: Any,
    prompt: str,
    max_tokens: Optional[int],
    temperature: Optional[float],
    stop: Optional[List[str]],
) -> None:
    """Worker function that runs generate() in a subprocess.

    Puts either the result or an exception info tuple on the queue.
    """
    try:
        result = backend.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        queue.put(("success", result))
    except Exception as e:
        # Can't pickle full exception, send type and message
        queue.put(("error", type(e).__name__, str(e)))


def _is_cuda_available() -> bool:
    """Check if CUDA is available for the loaded model."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def generate_with_timeout(
    backend: Any,
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    stop: Optional[List[str]] = None,
    timeout_seconds: Optional[int] = None,
) -> str:
    """Run backend.generate() with a timeout.

    If the generate() call does not complete within the timeout,
    raises GenerationTimeout. This prevents indefinite hangs.

    Parameters
    ----------
    backend : InferenceBackend
        The loaded inference backend
    prompt : str
        The prompt to generate from
    max_tokens : int, optional
        Maximum tokens to generate
    temperature : float, optional
        Generation temperature
    stop : list of str, optional
        Stop sequences
    timeout_seconds : int, optional
        Timeout in seconds (default: ESM_GEN_TIMEOUT env var or 300)

    Returns
    -------
    str
        The generated text

    Raises
    ------
    GenerationTimeout
        If the generate() call exceeds the timeout
    GenerationError
        If generate() fails with an error
    """
    if timeout_seconds is None:
        # Use shorter timeout for CPU (since max_tokens is also limited)
        if _is_cuda_available():
            timeout_seconds = DEFAULT_TIMEOUT_SECONDS
        else:
            timeout_seconds = CPU_TIMEOUT_SECONDS
            logger.debug(f"Using CPU timeout: {timeout_seconds}s")

    if not TIMEOUT_ENABLED:
        # Timeout disabled - run directly
        logger.debug("Generation timeout disabled, running directly")
        return backend.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )

    # IMPORTANT: Do NOT use multiprocessing on CPU by default!
    # Multiprocessing spawns a new process that reloads the 7GB model,
    # doubling memory usage and causing OOM on CI runners with 14-15GB RAM.
    #
    # Instead, use SIGALRM (Unix) or threading (Windows) timeout.
    # These cannot truly interrupt blocked C library calls (PyTorch CPU ops),
    # but they won't cause OOM. The shell-level timeout is the final safety net.
    #
    # Only use multiprocessing if explicitly forced via ESM_GEN_USE_MP=1
    use_mp = FORCE_MP_TIMEOUT  # Only if explicitly requested

    if use_mp and sys.platform != 'win32':
        # Multiprocessing - only when forced (will reload model, use more memory)
        logger.debug("Using multiprocessing timeout (explicitly forced)")
        return _generate_with_multiprocessing(
            backend, prompt, max_tokens, temperature, stop, timeout_seconds
        )
    elif hasattr(signal, 'SIGALRM'):
        # SIGALRM approach - won't interrupt blocked C calls on CPU, but won't OOM
        logger.debug("Using SIGALRM timeout (shell timeout is final safety net)")
        return _generate_with_alarm(
            backend, prompt, max_tokens, temperature, stop, timeout_seconds
        )
    else:
        # Windows fallback - threading timeout
        return _generate_with_thread_timeout(
            backend, prompt, max_tokens, temperature, stop, timeout_seconds
        )


def _worker_generate_spawn(
    backend_config_dict: Dict[str, Any],
    prompt: str,
    max_tokens: Optional[int],
    temperature: Optional[float],
    stop: Optional[List[str]],
    result_queue: Queue,
) -> None:
    """Worker function that runs in a spawned subprocess.

    Loads the model fresh in the subprocess to avoid fork issues with PyTorch.
    This is slower (model reload) but safe and reliable.
    """
    import traceback
    try:
        # Import inside worker for clean interpreter state
        from .inference import ModelConfig, TransformersBackend

        # Reconstruct model config from serialized dict
        model_config = ModelConfig.from_dict(backend_config_dict)

        # Load model in subprocess
        backend = TransformersBackend(model_config)
        backend.load()

        # Generate
        result = backend.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        result_queue.put(("success", result))

        # Clean up
        try:
            backend.unload()
        except Exception:
            pass

    except Exception as e:
        result_queue.put(("error", type(e).__name__, f"{e}\n{traceback.format_exc()}"))


def _serialize_backend_config(backend: Any) -> Dict[str, Any]:
    """Serialize backend config to a dict for passing to spawned worker."""
    config = getattr(backend, "config", None)
    if config is None:
        raise ValueError("Backend has no config attribute")

    # Convert dataclass/object to dict
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    elif hasattr(config, "_asdict"):  # namedtuple
        return config._asdict()
    else:
        raise ValueError(f"Cannot serialize backend config: {type(config)}")


def _generate_with_multiprocessing(
    backend: Any,
    prompt: str,
    max_tokens: Optional[int],
    temperature: Optional[float],
    stop: Optional[List[str]],
    timeout_seconds: int,
) -> str:
    """Generate with multiprocessing timeout (for CPU-bound operations).

    Uses 'spawn' to create a fresh interpreter process that loads the model
    cleanly. This avoids fork issues with PyTorch (thread duplication, deadlocks).

    The worker reloads the model in the subprocess, which is slower but safe
    and works reliably on GitHub Actions and other CI environments.

    This is more reliable than SIGALRM for CPU-bound operations because
    SIGALRM cannot interrupt blocked C library calls.
    """
    # Use spawn for clean interpreter (avoids PyTorch fork issues)
    ctx = multiprocessing.get_context('spawn')
    result_queue = ctx.Queue()

    # Serialize backend config for IPC
    try:
        backend_config_dict = _serialize_backend_config(backend)
    except Exception as e:
        logger.error(f"Failed to serialize backend config: {e}")
        raise GenerationError(f"Cannot use multiprocessing timeout: {e}")

    logger.debug(f"Starting generate() in spawned subprocess with {timeout_seconds}s timeout")
    t0 = time.time()

    proc = ctx.Process(
        target=_worker_generate_spawn,
        args=(backend_config_dict, prompt, max_tokens, temperature, stop, result_queue),
    )
    proc.start()

    # Wait for result with timeout
    proc.join(timeout=timeout_seconds)

    if proc.is_alive():
        # Process still running - timeout exceeded
        logger.warning(
            f"generate() exceeded {timeout_seconds}s timeout, terminating subprocess"
        )
        proc.terminate()
        proc.join(timeout=5)  # Give it 5s to terminate gracefully

        if proc.is_alive():
            # Still alive - force kill
            logger.warning("Process did not terminate, sending SIGKILL")
            proc.kill()
            proc.join(timeout=5)

        raise GenerationTimeout(
            f"model.generate() exceeded {timeout_seconds}s timeout (terminated)"
        )

    # Process finished - check result
    elapsed = time.time() - t0
    logger.debug(f"Subprocess completed in {elapsed:.1f}s")

    if result_queue.empty():
        raise GenerationError("Worker process died without producing result")

    status, *data = result_queue.get_nowait()

    if status == "success":
        logger.debug(f"generate() completed in {elapsed:.1f}s (under {timeout_seconds}s timeout)")
        return data[0]
    elif status == "error":
        exc_type, exc_msg = data
        raise GenerationError(f"Generation failed in subprocess: {exc_type}: {exc_msg}")
    else:
        raise GenerationError(f"Unknown status from worker: {status}")


def _generate_with_alarm(
    backend: Any,
    prompt: str,
    max_tokens: Optional[int],
    temperature: Optional[float],
    stop: Optional[List[str]],
    timeout_seconds: int,
) -> str:
    """Generate with Unix SIGALRM timeout.

    This works well for GPU operations where PyTorch periodically checks
    for signals. NOT effective for CPU-bound operations where C library
    calls block signal delivery.
    """
    def timeout_handler(signum, frame):
        raise GenerationTimeout(
            f"model.generate() exceeded {timeout_seconds}s timeout"
        )

    # Set up the alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        logger.debug(f"Starting generate() with {timeout_seconds}s alarm")
        t0 = time.time()

        result = backend.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )

        elapsed = time.time() - t0
        logger.debug(f"generate() completed in {elapsed:.1f}s (under {timeout_seconds}s timeout)")

        return result
    finally:
        # Cancel the alarm and restore old handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _generate_with_thread_timeout(
    backend: Any,
    prompt: str,
    max_tokens: Optional[int],
    temperature: Optional[float],
    stop: Optional[List[str]],
    timeout_seconds: int,
) -> str:
    """Generate with threading timeout (Windows fallback).

    Note: This cannot truly interrupt a blocked C call, but provides
    best-effort timeout detection.
    """
    import threading

    result_container = {"result": None, "error": None}

    def worker():
        try:
            result_container["result"] = backend.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
            )
        except Exception as e:
            result_container["error"] = e

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        # Thread still running - timeout exceeded
        # Note: We cannot forcefully kill the thread, but we can raise
        logger.error(
            f"generate() still running after {timeout_seconds}s - timeout exceeded. "
            f"Thread cannot be killed; process may need manual termination."
        )
        raise GenerationTimeout(
            f"model.generate() exceeded {timeout_seconds}s timeout"
        )

    if result_container["error"]:
        raise result_container["error"]

    return result_container["result"]


def check_memory_available(min_gb: float = 4.0) -> tuple[bool, float, str]:
    """Check if sufficient memory is available for generation.

    Parameters
    ----------
    min_gb : float
        Minimum required free memory in GB

    Returns
    -------
    tuple of (bool, float, str)
        (is_ok, available_gb, message)
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)

        if available_gb < min_gb:
            msg = (
                f"Low memory warning: {available_gb:.1f}GB available "
                f"(< {min_gb:.1f}GB required). Total: {total_gb:.1f}GB, "
                f"Used: {(mem.used / (1024 ** 3)):.1f}GB ({mem.percent}%)"
            )
            return False, available_gb, msg
        else:
            msg = f"Memory OK: {available_gb:.1f}GB available ({mem.percent}% used)"
            return True, available_gb, msg
    except ImportError:
        # psutil not available, try /proc/meminfo on Linux
        try:
            with open('/proc/meminfo') as f:
                meminfo = dict(
                    line.split(':') for line in f.read().strip().split('\n')
                    if ':' in line
                )
            available_kb = int(meminfo.get('MemAvailable', '0').strip().split()[0])
            available_gb = available_kb / (1024 ** 2)

            if available_gb < min_gb:
                msg = f"Low memory: {available_gb:.1f}GB available (< {min_gb:.1f}GB required)"
                return False, available_gb, msg
            else:
                msg = f"Memory OK: {available_gb:.1f}GB available"
                return True, available_gb, msg
        except Exception:
            return True, -1, "Memory check unavailable (no psutil or /proc/meminfo)"
    except Exception as e:
        return True, -1, f"Memory check failed: {e}"


def log_system_resources() -> Dict[str, Any]:
    """Log current system resources for debugging.

    Returns a dict with resource info that can be included in logs/traces.
    """
    info = {
        "timestamp": time.time(),
        "memory": {},
        "cpu": {},
    }

    try:
        import psutil

        # Memory
        mem = psutil.virtual_memory()
        info["memory"] = {
            "total_gb": round(mem.total / (1024 ** 3), 2),
            "available_gb": round(mem.available / (1024 ** 3), 2),
            "used_percent": mem.percent,
        }

        # Swap
        swap = psutil.swap_memory()
        info["memory"]["swap_used_gb"] = round(swap.used / (1024 ** 3), 2)
        info["memory"]["swap_percent"] = swap.percent

        # CPU
        info["cpu"]["percent"] = psutil.cpu_percent(interval=0.1)
        info["cpu"]["count"] = psutil.cpu_count()

        # Process-specific
        proc = psutil.Process()
        info["process"] = {
            "memory_gb": round(proc.memory_info().rss / (1024 ** 3), 2),
            "cpu_percent": proc.cpu_percent(),
        }

    except ImportError:
        # Try basic /proc stats
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        info["memory"]["total_gb"] = int(line.split()[1]) / (1024 ** 2)
                    elif line.startswith('MemAvailable:'):
                        info["memory"]["available_gb"] = int(line.split()[1]) / (1024 ** 2)
        except Exception:
            pass
    except Exception as e:
        info["error"] = str(e)

    return info
