"""Signer port definitions."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class SignerPort(Protocol):
    """Port for artifact signing implementations."""

    signer_id: str

    def sign_bytes(self, data: bytes) -> bytes:
        """Sign raw bytes and return a detached signature."""

    def sign_file(
        self,
        filepath: Path,
        output_path: Optional[Path] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Sign a file and optionally write a signature artifact."""
