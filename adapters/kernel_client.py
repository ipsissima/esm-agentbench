"""Kernel adapter backed by the certificates kernel client."""
from __future__ import annotations

from typing import Any, Dict, Optional

from certificates import kernel_client
from ports.kernel import KernelPort


class KernelClientAdapter(KernelPort):
    """Adapter that delegates to certificates.kernel_client."""

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
