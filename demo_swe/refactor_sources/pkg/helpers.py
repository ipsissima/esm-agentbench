
import math


def normalize_vector(vec):
    """Return a safely normalized vector, avoiding zero-division."""

    total = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / total for v in vec]
