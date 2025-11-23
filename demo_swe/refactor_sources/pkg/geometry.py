"""Test helpers to support refactor episode."""
from __future__ import annotations

from typing import Iterable, List

from pkg.helpers import normalize_vector


def compute_projection(vec: Iterable[float], direction: Iterable[float]) -> List[float]:
    """Project ``vec`` onto ``direction`` preserving the original magnitude."""

    base = list(vec)
    direction_unit = normalize_vector(direction)
    if not base or not direction_unit:
        return [0.0 for _ in base]
    magnitude = sum(v * v for v in base) ** 0.5
    return [magnitude * d for d in direction_unit]
