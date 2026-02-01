# esmassessor/kernel_adapter.py
"""Kernel Compute Port Adapter for verified kernel operations.

Provides a clean abstraction over the verified Coq/OCaml kernel and Python fallback.
This centralizes kernel loading, fallback logic, and error handling into a single
interface that the rest of the codebase can use without worrying about kernel
availability or repeated fallback warnings.

Architecture:
- KernelAdapterBase: Abstract interface implementing ports.kernel_compute.KernelComputePort
  for kernel computation operations (compute_residual, compute_bound, compute_certificate).
  This is a different abstraction from ports.kernel.KernelClientPort, which defines the
  interface for kernel process execution (run_kernel).
- VerifiedKernelAdapter: Uses the formally verified Coq/OCaml kernel
- PythonKernelAdapter: Pure Python fallback (not verified)
- make_kernel_adapter(): Factory that returns the best available adapter

Note: This module defines computation operations, not kernel process execution.
For kernel execution interfaces, see ports.kernel.KernelClientPort and adapters.kernel_client.
"""
from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from ports.kernel_compute import KernelComputePort

logger = logging.getLogger(__name__)

# Thread-safe initialization lock
_adapter_lock = threading.Lock()


class KernelAdapterError(RuntimeError):
    """Raised when kernel adapter encounters an unrecoverable error."""
    pass


class KernelAdapterBase(KernelComputePort, ABC):
    """Abstract interface for kernel computation operations.

    This implements the ports.kernel_compute.KernelComputePort protocol,
    defining the contract for computing spectral certificate metrics
    (residual, bound, certificate). Implementations may use the verified
    Coq/OCaml kernel or pure Python fallback.

    Note: This is the computation-level adapter. The process-level client
    port is ports.kernel.KernelClientPort (run_kernel/run_kernel_and_verify).
    KernelAdapterBase focuses on computation operations within the certificate
    generation pipeline.
    """
    """

    @abstractmethod
    def compute_residual(self, X0: np.ndarray, X1: np.ndarray, A: np.ndarray) -> float:
        """Compute residual using the kernel.

        Computes: residual = ||X1 - A @ X0||_F / ||X1||_F

        Parameters
        ----------
        X0 : np.ndarray
            State matrix at time t, shape (r, T-1).
        X1 : np.ndarray
            State matrix at time t+1, shape (r, T-1).
        A : np.ndarray
            Learned operator, shape (r, r).

        Returns
        -------
        float
            The residual error.

        Raises
        ------
        KernelAdapterError
            If computation fails.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_bound(
        self,
        residual: float,
        tail_energy: float,
        semantic_divergence: float,
        lipschitz_margin: float,
    ) -> float:
        """Compute theoretical bound using the kernel.

        Computes: bound = c_res*residual + c_tail*tail_energy
                        + c_sem*semantic_divergence + c_robust*lipschitz_margin

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

        Raises
        ------
        KernelAdapterError
            If computation fails.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_certificate(
        self,
        X0: np.ndarray,
        X1: np.ndarray,
        A: np.ndarray,
        tail_energy: float,
        semantic_divergence: float,
        lipschitz_margin: float,
    ) -> Tuple[float, float]:
        """Compute both residual and bound in a single call.

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

        Raises
        ------
        KernelAdapterError
            If computation fails.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_verified(self) -> bool:
        """Return True if this adapter uses the formally verified kernel."""
        raise NotImplementedError


class VerifiedKernelAdapter(KernelAdapterBase):
    """Adapter that uses the formally verified Coq/OCaml kernel.

    This adapter wraps the verified_kernel module and provides a clean
    interface for kernel operations. It fails fast if the kernel is not
    available rather than silently falling back.
    """

    def __init__(self, kernel_path: Optional[str] = None):
        """Initialize the verified kernel adapter.

        Parameters
        ----------
        kernel_path : str, optional
            Path to the kernel shared library. If not provided, uses
            VERIFIED_KERNEL_PATH environment variable or default locations.

        Raises
        ------
        KernelAdapterError
            If the verified kernel cannot be loaded.
        """
        # Import the verified kernel module
        try:
            from certificates import verified_kernel as vk
            self._vk = vk
        except ImportError as exc:
            raise KernelAdapterError(f"certificates.verified_kernel not importable: {exc}") from exc

        # Set environment variable if path provided
        if kernel_path:
            os.environ["VERIFIED_KERNEL_PATH"] = kernel_path

        # Attempt to load the kernel (strict mode - fail if unavailable)
        try:
            self._kernel = self._vk.load_kernel(strict=True)
        except self._vk.KernelError as exc:
            raise KernelAdapterError(f"Verified kernel not available: {exc}") from exc
        except Exception as exc:
            raise KernelAdapterError(f"Failed to load verified kernel: {exc}") from exc

        if self._kernel is None:
            raise KernelAdapterError("Verified kernel not found or failed to load")

        logger.info("VerifiedKernelAdapter initialized successfully")

    @property
    def is_verified(self) -> bool:
        return True

    def compute_residual(self, X0: np.ndarray, X1: np.ndarray, A: np.ndarray) -> float:
        try:
            return self._vk.compute_residual(X0, X1, A, strict=True)
        except self._vk.KernelError as exc:
            raise KernelAdapterError(f"Residual computation failed: {exc}") from exc
        except Exception as exc:
            raise KernelAdapterError(f"Unexpected error in compute_residual: {exc}") from exc

    def compute_bound(
        self,
        residual: float,
        tail_energy: float,
        semantic_divergence: float,
        lipschitz_margin: float,
    ) -> float:
        try:
            return self._vk.compute_bound(
                residual, tail_energy, semantic_divergence, lipschitz_margin, strict=True
            )
        except self._vk.KernelError as exc:
            raise KernelAdapterError(f"Bound computation failed: {exc}") from exc
        except Exception as exc:
            raise KernelAdapterError(f"Unexpected error in compute_bound: {exc}") from exc

    def compute_certificate(
        self,
        X0: np.ndarray,
        X1: np.ndarray,
        A: np.ndarray,
        tail_energy: float,
        semantic_divergence: float,
        lipschitz_margin: float,
    ) -> Tuple[float, float]:
        try:
            return self._vk.compute_certificate(
                X0, X1, A, tail_energy, semantic_divergence, lipschitz_margin, strict=True
            )
        except self._vk.KernelError as exc:
            raise KernelAdapterError(f"Certificate computation failed: {exc}") from exc
        except Exception as exc:
            raise KernelAdapterError(f"Unexpected error in compute_certificate: {exc}") from exc

    def selftest(self, n: int = 2, r: int = 4, tol: float = 1e-6) -> bool:
        """Run a self-test to verify kernel is working correctly.

        Parameters
        ----------
        n : int
            Number of time steps for test matrices.
        r : int
            Rank/dimension of test matrices.
        tol : float
            Tolerance for numerical checks.

        Returns
        -------
        bool
            True if self-test passes.

        Raises
        ------
        KernelAdapterError
            If self-test fails.
        """
        # Generate random test data
        np.random.seed(42)  # Reproducible test
        X0 = np.random.randn(r, n).astype(np.float64)
        A = np.random.randn(r, r).astype(np.float64)
        X1 = A @ X0 + 0.001 * np.random.randn(r, n).astype(np.float64)

        # Compute certificate
        res, bound = self.compute_certificate(X0, X1, A, 0.01, 0.01, 0.01)

        # Basic sanity checks
        if not isinstance(res, (int, float)):
            raise KernelAdapterError(f"Self-test failed: residual not numeric ({type(res)})")
        if not isinstance(bound, (int, float)):
            raise KernelAdapterError(f"Self-test failed: bound not numeric ({type(bound)})")
        if res < 0:
            raise KernelAdapterError(f"Self-test failed: residual negative ({res})")
        if bound < 0:
            raise KernelAdapterError(f"Self-test failed: bound negative ({bound})")

        logger.debug("VerifiedKernelAdapter self-test passed: res=%.6f, bound=%.6f", res, bound)
        return True


class PythonKernelAdapter(KernelAdapterBase):
    """Pure Python adapter for kernel operations.

    This adapter provides the same interface as VerifiedKernelAdapter but
    uses pure Python implementations. It is NOT formally verified and should
    only be used as a fallback when the verified kernel is unavailable.
    """

    def __init__(self):
        """Initialize the Python kernel adapter."""
        # Import the verified kernel module for Python fallback functions
        try:
            from certificates import verified_kernel as vk
            self._vk = vk
        except ImportError as exc:
            raise KernelAdapterError(f"certificates.verified_kernel not importable: {exc}") from exc

        logger.warning(
            "PythonKernelAdapter initialized - using UNVERIFIED Python fallback. "
            "Certificate computations are not formally verified."
        )

    @property
    def is_verified(self) -> bool:
        return False

    def compute_residual(self, X0: np.ndarray, X1: np.ndarray, A: np.ndarray) -> float:
        try:
            return self._vk.compute_residual(X0, X1, A, strict=False)
        except Exception as exc:
            raise KernelAdapterError(f"Python residual computation failed: {exc}") from exc

    def compute_bound(
        self,
        residual: float,
        tail_energy: float,
        semantic_divergence: float,
        lipschitz_margin: float,
    ) -> float:
        try:
            return self._vk.compute_bound(
                residual, tail_energy, semantic_divergence, lipschitz_margin, strict=False
            )
        except Exception as exc:
            raise KernelAdapterError(f"Python bound computation failed: {exc}") from exc

    def compute_certificate(
        self,
        X0: np.ndarray,
        X1: np.ndarray,
        A: np.ndarray,
        tail_energy: float,
        semantic_divergence: float,
        lipschitz_margin: float,
    ) -> Tuple[float, float]:
        # Use the individual computations with strict=False
        residual = self.compute_residual(X0, X1, A)
        bound = self.compute_bound(residual, tail_energy, semantic_divergence, lipschitz_margin)
        return (residual, bound)

    def selftest(self, n: int = 2, r: int = 4, tol: float = 1e-6) -> bool:
        """Run a self-test to verify Python fallback is working correctly.

        Parameters
        ----------
        n : int
            Number of time steps for test matrices.
        r : int
            Rank/dimension of test matrices.
        tol : float
            Tolerance for numerical checks.

        Returns
        -------
        bool
            True if self-test passes.

        Raises
        ------
        KernelAdapterError
            If self-test fails.
        """
        # Generate random test data
        np.random.seed(42)  # Reproducible test
        X0 = np.random.randn(r, n).astype(np.float64)
        A = np.random.randn(r, r).astype(np.float64)
        X1 = A @ X0 + 0.001 * np.random.randn(r, n).astype(np.float64)

        # Compute certificate
        res, bound = self.compute_certificate(X0, X1, A, 0.01, 0.01, 0.01)

        # Basic sanity checks
        if not isinstance(res, (int, float)):
            raise KernelAdapterError(f"Self-test failed: residual not numeric ({type(res)})")
        if not isinstance(bound, (int, float)):
            raise KernelAdapterError(f"Self-test failed: bound not numeric ({type(bound)})")
        if res < 0:
            raise KernelAdapterError(f"Self-test failed: residual negative ({res})")
        if bound < 0:
            raise KernelAdapterError(f"Self-test failed: bound negative ({bound})")

        logger.debug("PythonKernelAdapter self-test passed: res=%.6f, bound=%.6f", res, bound)
        return True


# Note: This module defines computation abstractions, distinct from the kernel
# execution protocol defined in ports.kernel.KernelPort. KernelAdapterBase focuses
# on certificate computation operations (compute_residual, compute_bound, compute_certificate),
# while ports.KernelPort defines the interface for kernel execution (run_kernel methods).
# See adapters.kernel_client.KernelClientAdapter for an implementation of ports.KernelPort.


# Global adapter instance (lazy initialized)
_kernel_adapter: Optional[KernelAdapterBase] = None
_adapter_init_attempted: bool = False


def make_kernel_adapter(prefer_verified: bool = True) -> KernelAdapterBase:
    """Factory function to create the best available kernel adapter.

    This function is thread-safe and attempts to create a VerifiedKernelAdapter
    first, falling back to PythonKernelAdapter if the verified kernel is unavailable.

    Parameters
    ----------
    prefer_verified : bool
        If True, try to use the verified kernel first. If False, use Python
        adapter directly (useful for testing).

    Returns
    -------
    KernelAdapterBase
        The best available kernel adapter.
    """
    global _kernel_adapter, _adapter_init_attempted

    # Fast path: return cached adapter without lock
    if _kernel_adapter is not None:
        return _kernel_adapter

    # Slow path: acquire lock for initialization
    with _adapter_lock:
        # Double-check after acquiring lock (another thread may have initialized)
        if _kernel_adapter is not None:
            return _kernel_adapter

        # Avoid repeated initialization attempts
        if _adapter_init_attempted:
            # Fall back to Python adapter if we already tried and failed
            _kernel_adapter = PythonKernelAdapter()
            return _kernel_adapter

        _adapter_init_attempted = True

        if prefer_verified:
            try:
                _kernel_adapter = VerifiedKernelAdapter()
                logger.info("Using verified kernel adapter")
                return _kernel_adapter
            except KernelAdapterError as exc:
                logger.warning(
                    "Verified kernel unavailable, using Python fallback: %s", exc
                )

        _kernel_adapter = PythonKernelAdapter()
        return _kernel_adapter


def get_kernel_adapter() -> KernelAdapterBase:
    """Get the current kernel adapter, initializing if necessary.

    Returns
    -------
    KernelAdapterBase
        The current kernel adapter.
    """
    global _kernel_adapter
    if _kernel_adapter is None:
        return make_kernel_adapter()
    return _kernel_adapter


def reset_kernel_adapter() -> None:
    """Reset the kernel adapter (mainly for testing).

    This clears the cached adapter and allows re-initialization.
    Thread-safe.
    """
    global _kernel_adapter, _adapter_init_attempted
    with _adapter_lock:
        _kernel_adapter = None
        _adapter_init_attempted = False


__all__ = [
    "KernelAdapterBase",
    "KernelAdapterError",
    "VerifiedKernelAdapter",
    "PythonKernelAdapter",
    "make_kernel_adapter",
    "get_kernel_adapter",
    "reset_kernel_adapter",
]
