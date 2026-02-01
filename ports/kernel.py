"""Kernel client port definitions.

Defines the KernelClientPort protocol for kernel execution operations.

This is the process-level kernel interface for executing the verified
Coq/OCaml kernel as a subprocess and parsing its output.

Note: This is distinct from ports.kernel_compute.KernelComputePort,
which provides computation operations (compute_residual, compute_bound, compute_certificate).
KernelClientPort focuses on kernel process execution interfaces (run_kernel methods).

See adapters.kernel_client.KernelClientAdapter for a concrete implementation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class KernelClientPort(Protocol):
    """Port for verified kernel client execution.

    This protocol defines the interface for kernel execution operations,
    which run the verified Coq/OCaml kernel as a subprocess and parse its output.

    This is the canonical kernel process-level execution interface; concrete
    adapters should implement this protocol. See adapters.kernel_client.KernelClientAdapter
    for the standard implementation.

    Note: This is distinct from ports.kernel_compute.KernelComputePort,
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
