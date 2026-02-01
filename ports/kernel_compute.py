"""Kernel computation port definitions.

Defines the KernelComputePort protocol for kernel computation operations.

This is the computation-level port. Implementations compute residuals,
theoretical bounds and full certificate values using either a verified
kernel or a Python fallback.

This is distinct from ports.kernel.KernelClientPort, which defines the
process-level interface for kernel execution (run_kernel methods).

See esmassessor.kernel_adapter.KernelAdapterBase for concrete implementations.
"""
from __future__ import annotations

from typing import Protocol, Tuple, runtime_checkable

import numpy as np


@runtime_checkable
class KernelComputePort(Protocol):
    """Protocol for adapters that perform kernel computations.

    This is the computation-level port. Implementations compute residuals,
    theoretical bounds and full certificate values using either a verified
    kernel or a Python fallback.

    This interface is used throughout the certificate generation pipeline
    to compute spectral metrics without depending on a specific kernel
    implementation (verified vs Python fallback).
    """

    def compute_residual(self, X0: np.ndarray, X1: np.ndarray, A: np.ndarray) -> float:
        """Compute the Frobenius norm of the residual ||X1 - A @ X0||_F.

        Args:
            X0: Initial embedding matrix (n_samples, n_features)
            X1: Next-step embedding matrix (n_samples, n_features)
            A: Koopman/PCA transition matrix (n_features, n_features)

        Returns:
            Residual value (non-negative float)
        """
        ...

    def compute_bound(
        self,
        residual: float,
        tail_energy: float,
        semantic_divergence: float,
        lipschitz_margin: float,
    ) -> float:
        """Compute the theoretical bound for spectral certificate.

        Args:
            residual: Pre-computed Frobenius residual
            tail_energy: Energy in truncated singular values
            semantic_divergence: Semantic drift measure
            lipschitz_margin: Lipschitz constant margin

        Returns:
            Theoretical bound value (non-negative float)
        """
        ...

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

        This is the primary method for certificate generation. It computes
        both the residual and theoretical bound, which together form the
        spectral certificate.

        Args:
            X0: Initial embedding matrix (n_samples, n_features)
            X1: Next-step embedding matrix (n_samples, n_features)
            A: Koopman/PCA transition matrix (n_features, n_features)
            tail_energy: Energy in truncated singular values
            semantic_divergence: Semantic drift measure
            lipschitz_margin: Lipschitz constant margin

        Returns:
            Tuple of (residual, theoretical_bound)
        """
        ...

    @property
    def is_verified(self) -> bool:
        """Return True if this adapter uses a formally verified kernel.

        Returns:
            True for verified kernel implementations, False for Python fallback
        """
        ...
