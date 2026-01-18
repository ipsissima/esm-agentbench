# esmassessor/kernel_adapter.py
"""Kernel Port Adapter for verified kernel operations.

Provides a clean abstraction over the verified Coq/OCaml kernel and Python fallback.
This centralizes kernel loading, fallback logic, and error handling into a single
interface that the rest of the codebase can use without worrying about kernel
availability or repeated fallback warnings.

Architecture:
- KernelPort: Abstract interface for kernel operations
- VerifiedKernelAdapter: Uses the formally verified Coq/OCaml kernel
- PythonKernelAdapter: Pure Python fallback (not verified)
- make_kernel_adapter(): Factory that returns the best available adapter
"""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class KernelAdapterError(RuntimeError):
    """Raised when kernel adapter encounters an unrecoverable error."""
    pass


class KernelPort(ABC):
    """Abstract interface for kernel operations.

    This port defines the contract for computing spectral certificate metrics.
    Implementations may use the verified Coq/OCaml kernel or pure Python.
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


class VerifiedKernelAdapter(KernelPort):
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


class PythonKernelAdapter(KernelPort):
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


# Global adapter instance (lazy initialized)
_kernel_adapter: Optional[KernelPort] = None
_adapter_init_attempted: bool = False


def make_kernel_adapter(prefer_verified: bool = True) -> KernelPort:
    """Factory function to create the best available kernel adapter.

    This function attempts to create a VerifiedKernelAdapter first, and
    falls back to PythonKernelAdapter if the verified kernel is unavailable.

    Parameters
    ----------
    prefer_verified : bool
        If True, try to use the verified kernel first. If False, use Python
        adapter directly (useful for testing).

    Returns
    -------
    KernelPort
        The best available kernel adapter.
    """
    global _kernel_adapter, _adapter_init_attempted

    # Return cached adapter if available
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


def get_kernel_adapter() -> KernelPort:
    """Get the current kernel adapter, initializing if necessary.

    Returns
    -------
    KernelPort
        The current kernel adapter.
    """
    global _kernel_adapter
    if _kernel_adapter is None:
        return make_kernel_adapter()
    return _kernel_adapter


def reset_kernel_adapter() -> None:
    """Reset the kernel adapter (mainly for testing).

    This clears the cached adapter and allows re-initialization.
    """
    global _kernel_adapter, _adapter_init_attempted
    _kernel_adapter = None
    _adapter_init_attempted = False


__all__ = [
    "KernelPort",
    "KernelAdapterError",
    "VerifiedKernelAdapter",
    "PythonKernelAdapter",
    "make_kernel_adapter",
    "get_kernel_adapter",
    "reset_kernel_adapter",
]
