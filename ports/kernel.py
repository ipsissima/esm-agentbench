"""Kernel port definitions.

Defines the KernelPort protocol for kernel execution operations.

Note: This is distinct from esmassessor.kernel_adapter.KernelAdapterBase,
which provides computation operations (compute_residual, compute_bound, compute_certificate).
KernelPort focuses on kernel execution interfaces (run_kernel methods).

See adapters.kernel_client.KernelClientAdapter for a concrete implementation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class KernelPort(Protocol):
    """Port for verified kernel execution.

    This protocol defines the interface for kernel execution operations,
    which run the verified Coq/OCaml kernel and parse its output.

    This is the canonical kernel execution interface; concrete adapters should
    implement this protocol. See adapters.kernel_client.KernelClientAdapter
    for the standard implementation.

    Note: This is distinct from esmassessor.kernel_adapter.KernelAdapterBase,
    which provides computation abstractions (compute_residual, compute_bound).
    """

    def run_kernel(
        self,
        kernel_input_path: str,
        output_path: Optional[str] = None,
        precision_bits: int = 128,
        mode: str = "prototype",
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute the kernel and return parsed output."""

    def run_kernel_and_verify(
        self,
        kernel_input_path: str,
        output_path: Optional[str] = None,
        precision_bits: int = 128,
        mode: str = "prototype",
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute the kernel and verify output checks."""
