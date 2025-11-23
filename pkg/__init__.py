"""Proxy package exposing refactor helper modules from demo_swe/refactor_sources."""
from __future__ import annotations

from pathlib import Path

_refactor_pkg = Path(__file__).resolve().parent.parent / "demo_swe" / "refactor_sources" / "pkg"
__path__ = [str(_refactor_pkg)] + list(__path__)  # type: ignore[name-defined]

__all__ = []
