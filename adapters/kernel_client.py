"""Kernel adapter backed by the certificates kernel client.

Provides the standard implementation of ports.kernel.KernelClientPort, which defines
the interface for kernel execution operations. This adapter delegates to the
certificates.kernel_client module for actual kernel invocation.

Note: This adapter implements kernel process execution (run_kernel methods), not
computation operations. For computation abstractions, see
ports.kernel_compute.KernelComputePort and esmassessor.kernel_adapter.KernelAdapterBase.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from certificates import kernel_client
from ports.kernel import KernelClientPort


class KernelClientAdapter(KernelClientPort):
    """Adapter that delegates to certificates.kernel_client.

    This is the standard implementation of the KernelClientPort protocol, providing
    kernel execution operations by delegating to the certificates.kernel_client
    module. Use this adapter when you need to invoke the verified Coq/OCaml
    kernel for certificate verification.
    """

    def run_kernel(
        self,
        kernel_input_path: str,
        output_path: Optional[str] = None,
        precision_bits: int = 128,
        mode: str = "prototype",
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        return kernel_client.run_kernel(
            kernel_input_path,
            output_path=output_path,
            precision_bits=precision_bits,
            mode=mode,
            timeout=timeout,
        )

    def run_kernel_and_verify(
        self,
        kernel_input_path: str,
        output_path: Optional[str] = None,
        precision_bits: int = 128,
        mode: str = "prototype",
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        return kernel_client.run_kernel_and_verify(
            kernel_input_path,
            output_path=output_path,
            precision_bits=precision_bits,
            mode=mode,
            timeout=timeout,
        )
