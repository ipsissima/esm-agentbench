"""Geometry helpers used by the refactor episode tests."""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from pkg.helpers import normalize_vector


def compute_projection(vec: Iterable[float], direction: Iterable[float]) -> List[float]:
    """Project ``vec`` onto ``direction`` preserving the original magnitude."""

    base = list(vec)
    direction_unit = normalize_vector(direction)
    if not base or not direction_unit:
        return [0.0 for _ in base]
    magnitude = sum(v * v for v in base) ** 0.5
    return [magnitude * d for d in direction_unit]


def projection_shape(points: Sequence[Sequence[float] | float]) -> Tuple[int, ...]:
    """Return a lightweight shape tuple for the provided data."""

    if not points:
        return (0,)
    first = points[0]
    if isinstance(first, (list, tuple)):
        inner_len = len(first)
        return (len(points), inner_len)
    return (len(points),)


__all__ = ["compute_projection", "projection_shape"]
