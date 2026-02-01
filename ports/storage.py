"""Trace storage port definitions."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, runtime_checkable


@runtime_checkable
class TraceStoragePort(Protocol):
    """Port for trace storage implementations."""

    def save_trace(
        self,
        metadata: Mapping[str, Any],
        steps: Sequence[Mapping[str, Any]],
        outcome: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        """Persist a trace and return its file path."""

    def load_trace(self, trace_file: Path) -> Dict[str, Any]:
        """Load a trace from storage."""
