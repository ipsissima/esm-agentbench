
from .helpers import normalize_vector


def compute_projection(u, v):
    """Project vector ``u`` onto ``v`` using helper normalization."""

    norm_v = normalize_vector(v)
    dot = sum(a * b for a, b in zip(u, norm_v))
    return [dot * b for b in norm_v]
