"""Proxy package exposing refactor helper modules from demo_swe/refactor_sources.

IMPORTANT: This module uses dynamic path manipulation to expose the pkg module
from demo_swe/refactor_sources/pkg for demonstration purposes. This is intentional
for the SWE-bench demo scenario but should not be used as a pattern elsewhere.

The demo_swe scenarios test the agent's ability to refactor code that imports
from 'pkg', so we need this proxy to make those imports work.
"""
from __future__ import annotations

from pathlib import Path

_refactor_pkg = Path(__file__).resolve().parent.parent / "demo_swe" / "refactor_sources" / "pkg"
if _refactor_pkg.exists():
    __path__ = [str(_refactor_pkg)] + list(__path__)  # type: ignore[name-defined]

__all__ = []
