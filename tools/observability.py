"""Runtime Safeguards and Observability.

This module provides:
- Structured logging for model operations
- Resource limits and monitoring
- Crash diagnostics and core dump capture
- Metrics collection for CI dashboards

Usage:
    from tools.observability import (
        log_generation_event,
        ResourceLimiter,
        CrashDiagnostics,
    )

    # Log a generation event
    log_generation_event(
        model="phi-3",
        use_cache=False,
        attn_impl="eager",
        elapsed_s=27.3,
    )

    # Apply resource limits
    with ResourceLimiter(max_memory_mb=8192, timeout_s=300):
        result = model.generate(...)

    # Capture diagnostics on crash
    diagnostics = CrashDiagnostics.capture()
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import resource
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Environment variables
ENV_LOG_JSON = "ESM_LOG_JSON"
ENV_METRICS_FILE = "ESM_METRICS_FILE"
ENV_ENABLE_DIAGNOSTICS = "ESM_ENABLE_DIAGNOSTICS"


@dataclass
class GenerationEvent:
    """Structured event for model generation."""
    timestamp: str
    model: str
    use_cache: bool
    attn_impl: str
    elapsed_s: float
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    success: bool = True
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    timestamp: str
    memory_mb: float
    cpu_percent: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class CrashReport:
    """Report generated on process crash or kernel failure."""
    timestamp: str
    error_type: str
    error_message: str
    traceback: str
    environment: Dict[str, str]
    ld_library_path: Optional[str] = None
    glibc_version: Optional[str] = None
    python_version: str = ""
    kernel_loaded: bool = False
    core_dump_path: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class MetricsCollector:
    """Collects and writes metrics to file for CI dashboards."""

    def __init__(self, output_path: Optional[str] = None):
        """Initialize metrics collector.

        Parameters
        ----------
        output_path : str, optional
            Path to write metrics. Uses ESM_METRICS_FILE env var if not provided.
        """
        self.output_path = output_path or os.environ.get(ENV_METRICS_FILE)
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def record_event(self, event: Union[GenerationEvent, ResourceMetrics, Dict]) -> None:
        """Record a metrics event."""
        with self._lock:
            if isinstance(event, (GenerationEvent, ResourceMetrics)):
                self._events.append(asdict(event))
            else:
                self._events.append(event)

    def flush(self) -> None:
        """Flush events to output file."""
        if not self.output_path:
            return

        with self._lock:
            if not self._events:
                return

            try:
                # Append to file (one JSON object per line)
                with open(self.output_path, "a") as f:
                    for event in self._events:
                        f.write(json.dumps(event) + "\n")
                self._events.clear()
            except Exception as exc:
                logger.error(f"Failed to flush metrics: {exc}")

    def __del__(self):
        """Ensure events are flushed on cleanup."""
        try:
            self.flush()
        except Exception:
            pass


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics_collector

    if _metrics_collector is not None:
        return _metrics_collector

    with _collector_lock:
        if _metrics_collector is not None:
            return _metrics_collector

        _metrics_collector = MetricsCollector()
        atexit.register(_metrics_collector.flush)
        return _metrics_collector


def log_generation_event(
    model: str,
    use_cache: bool,
    attn_impl: str,
    elapsed_s: float,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    success: bool = True,
    error: Optional[str] = None,
    **extra: Any,
) -> GenerationEvent:
    """Log a generation event.

    Parameters
    ----------
    model : str
        Model identifier.
    use_cache : bool
        Whether KV cache was used.
    attn_impl : str
        Attention implementation used.
    elapsed_s : float
        Elapsed time in seconds.
    input_tokens : int, optional
        Number of input tokens.
    output_tokens : int, optional
        Number of output tokens.
    success : bool
        Whether generation succeeded.
    error : str, optional
        Error message if generation failed.
    **extra
        Additional metadata.

    Returns
    -------
    GenerationEvent
        The logged event.
    """
    event = GenerationEvent(
        timestamp=datetime.utcnow().isoformat(),
        model=model,
        use_cache=use_cache,
        attn_impl=attn_impl,
        elapsed_s=elapsed_s,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        success=success,
        error=error,
        extra=extra,
    )

    # Log as JSON if enabled
    if os.environ.get(ENV_LOG_JSON, "0") == "1":
        logger.info(event.to_json())
    else:
        logger.info(
            "Generation: model=%s use_cache=%s attn_impl=%s elapsed=%.3fs success=%s",
            model, use_cache, attn_impl, elapsed_s, success
        )

    # Record in metrics collector
    get_metrics_collector().record_event(event)

    return event


class ResourceLimiter:
    """Context manager for applying resource limits.

    Provides protection against runaway memory usage and timeouts.
    """

    def __init__(
        self,
        max_memory_mb: Optional[int] = None,
        timeout_s: Optional[int] = None,
        max_cpu_s: Optional[int] = None,
    ):
        """Initialize resource limiter.

        Parameters
        ----------
        max_memory_mb : int, optional
            Maximum memory usage in MB.
        timeout_s : int, optional
            Timeout in seconds.
        max_cpu_s : int, optional
            Maximum CPU time in seconds.
        """
        self.max_memory_mb = max_memory_mb
        self.timeout_s = timeout_s
        self.max_cpu_s = max_cpu_s
        self._old_limits: Dict[int, tuple] = {}
        self._alarm_handler = None

    def __enter__(self) -> "ResourceLimiter":
        """Apply resource limits."""
        # Memory limit
        if self.max_memory_mb:
            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_AS)
                self._old_limits[resource.RLIMIT_AS] = (soft, hard)
                limit_bytes = self.max_memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard))
                logger.debug(f"Set memory limit: {self.max_memory_mb}MB")
            except (ValueError, OSError) as exc:
                logger.warning(f"Could not set memory limit: {exc}")

        # CPU time limit
        if self.max_cpu_s:
            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
                self._old_limits[resource.RLIMIT_CPU] = (soft, hard)
                resource.setrlimit(resource.RLIMIT_CPU, (self.max_cpu_s, hard))
                logger.debug(f"Set CPU time limit: {self.max_cpu_s}s")
            except (ValueError, OSError) as exc:
                logger.warning(f"Could not set CPU limit: {exc}")

        # Timeout via alarm signal
        if self.timeout_s:
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Operation timed out after {self.timeout_s}s")

            self._alarm_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout_s)
            logger.debug(f"Set timeout: {self.timeout_s}s")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Restore original limits."""
        # Restore alarm
        if self.timeout_s:
            signal.alarm(0)
            if self._alarm_handler:
                signal.signal(signal.SIGALRM, self._alarm_handler)

        # Restore resource limits
        for res_type, (soft, hard) in self._old_limits.items():
            try:
                resource.setrlimit(res_type, (soft, hard))
            except (ValueError, OSError):
                pass

        return False  # Don't suppress exceptions


class CrashDiagnostics:
    """Capture diagnostic information on crashes."""

    @classmethod
    def capture(
        cls,
        error: Optional[Exception] = None,
        include_env: bool = True,
    ) -> CrashReport:
        """Capture crash diagnostics.

        Parameters
        ----------
        error : Exception, optional
            The exception that caused the crash.
        include_env : bool
            Whether to include environment variables.

        Returns
        -------
        CrashReport
            Diagnostic report.
        """
        # Get traceback
        if error:
            tb = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            error_type = type(error).__name__
            error_message = str(error)
        else:
            tb = traceback.format_exc()
            error_type = "Unknown"
            error_message = "Unknown error"

        # Get environment (filter sensitive values)
        env = {}
        if include_env:
            sensitive_keys = {"PASSWORD", "SECRET", "TOKEN", "KEY", "CREDENTIAL"}
            for key, value in os.environ.items():
                if any(s in key.upper() for s in sensitive_keys):
                    env[key] = "***REDACTED***"
                else:
                    env[key] = value

        # Get system info
        ld_library_path = os.environ.get("LD_LIBRARY_PATH")

        glibc_version = None
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            glibc_version = libc.gnu_get_libc_version().decode()
        except Exception:
            pass

        # Check if kernel was loaded
        kernel_loaded = False
        try:
            from certificates import verified_kernel as vk
            kernel_loaded = vk._kernel_lib is not None
        except Exception:
            pass

        return CrashReport(
            timestamp=datetime.utcnow().isoformat(),
            error_type=error_type,
            error_message=error_message,
            traceback=tb,
            environment=env,
            ld_library_path=ld_library_path,
            glibc_version=glibc_version,
            python_version=sys.version,
            kernel_loaded=kernel_loaded,
        )

    @classmethod
    def save(cls, report: CrashReport, path: Optional[str] = None) -> str:
        """Save crash report to file.

        Parameters
        ----------
        report : CrashReport
            The crash report.
        path : str, optional
            Output path. If not provided, creates temp file.

        Returns
        -------
        str
            Path to saved report.
        """
        if path is None:
            fd, path = tempfile.mkstemp(suffix="_crash_report.json", prefix="esm_")
            os.close(fd)

        Path(path).write_text(report.to_json())
        logger.info(f"Crash report saved to: {path}")

        return path


def install_crash_handler() -> None:
    """Install global crash handler for unhandled exceptions."""
    def exception_handler(exc_type, exc_value, exc_tb):
        """Handle unhandled exceptions."""
        if os.environ.get(ENV_ENABLE_DIAGNOSTICS, "0") != "1":
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return

        try:
            report = CrashDiagnostics.capture(exc_value)
            path = CrashDiagnostics.save(report)
            logger.error(f"Crash report: {path}")
        except Exception:
            pass

        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = exception_handler
    logger.debug("Installed global crash handler")


@contextmanager
def capture_diagnostics_on_error():
    """Context manager that captures diagnostics on any error."""
    try:
        yield
    except Exception as exc:
        if os.environ.get(ENV_ENABLE_DIAGNOSTICS, "0") == "1":
            report = CrashDiagnostics.capture(exc)
            path = CrashDiagnostics.save(report)
            logger.error(f"Error diagnostics saved: {path}")
        raise


def get_current_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback using /proc
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024  # KB to MB
        except Exception:
            pass
    return 0.0


def get_gpu_metrics() -> Optional[Dict[str, float]]:
    """Get GPU memory and utilization metrics."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None

        return {
            "memory_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
            "memory_reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
            "max_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        }
    except ImportError:
        return None


def log_resource_metrics() -> ResourceMetrics:
    """Log current resource metrics."""
    gpu_metrics = get_gpu_metrics()

    metrics = ResourceMetrics(
        timestamp=datetime.utcnow().isoformat(),
        memory_mb=get_current_memory_mb(),
        cpu_percent=0.0,  # Would need psutil for accurate CPU %
        gpu_memory_mb=gpu_metrics.get("memory_allocated_mb") if gpu_metrics else None,
        gpu_utilization=None,
    )

    if os.environ.get(ENV_LOG_JSON, "0") == "1":
        logger.info(metrics.to_json())
    else:
        logger.info(
            "Resources: memory=%.1fMB gpu_memory=%s",
            metrics.memory_mb,
            f"{metrics.gpu_memory_mb:.1f}MB" if metrics.gpu_memory_mb else "N/A"
        )

    get_metrics_collector().record_event(metrics)
    return metrics


__all__ = [
    # Events
    "GenerationEvent",
    "ResourceMetrics",
    "CrashReport",
    # Functions
    "log_generation_event",
    "log_resource_metrics",
    "get_current_memory_mb",
    "get_gpu_metrics",
    # Classes
    "MetricsCollector",
    "ResourceLimiter",
    "CrashDiagnostics",
    # Utilities
    "get_metrics_collector",
    "install_crash_handler",
    "capture_diagnostics_on_error",
]
