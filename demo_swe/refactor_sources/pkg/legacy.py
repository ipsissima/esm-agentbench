"""Legacy shims so refactor tests can import stable symbols."""
from __future__ import annotations

from pkg.geometry import compute_projection, projection_shape

__all__ = ["compute_projection", "projection_shape"]
