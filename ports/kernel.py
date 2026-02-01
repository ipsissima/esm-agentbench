"""Kernel port definitions."""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class KernelPort(Protocol):
    """Port for verified kernel execution.

    This is the canonical kernel interface; concrete adapters should implement
    this protocol rather than redefining a separate contract.
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
