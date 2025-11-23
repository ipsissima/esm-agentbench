"""Utility helpers used by the refactor episode tests."""
from __future__ import annotations

import math
from typing import Iterable, List


def normalize_vector(vec: Iterable[float]) -> List[float]:
    """Return a unit-length copy of the vector, defaulting to zeros on empty."""

    vals = [float(x) for x in vec]
    if not vals:
        return []
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


def normalize(vec: Iterable[float]) -> List[float]:
    """Alias for normalize_vector for backward compatibility."""

    return normalize_vector(vec)


__all__ = ["normalize_vector", "normalize"]
