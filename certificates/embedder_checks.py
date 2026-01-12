import numpy as np


class DegenerateEmbeddingError(RuntimeError):
    """Raised when embeddings collapse to near-constant vectors."""


def normalize_and_check_embeddings(
    embs: np.ndarray,
    per_step_norm: bool = True,
    var_threshold: float = 1e-6,
    time_delay_m: int = 2,
) -> np.ndarray:
    """Normalize embeddings and detect collapsed representations.

    Embeddings are expected as a 2D (T, d) array. If per-step normalization
    is enabled, each row is L2-normalized. Collapse is detected via mean
    per-dimension variance. When collapsed, attempt a time-delay stack;
    raise DegenerateEmbeddingError if still collapsed.
    """
    X = np.asarray(embs, dtype=float)
    if X.ndim != 2:
        raise DegenerateEmbeddingError("Embeddings must be 2D (T x d)")
    if per_step_norm:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / (norms + 1e-12)
    mean_var = float(np.var(X, axis=0).mean())
    if mean_var >= var_threshold:
        return X
    T, _ = X.shape
    if T < time_delay_m + 1:
        raise DegenerateEmbeddingError(
            f"Embeddings collapsed (mean_var={mean_var:.3e}) and trace too short for time-delay"
        )
    stacked_rows = []
    for t in range(time_delay_m - 1, T):
        stacked_rows.append(np.hstack([X[t - i] for i in range(time_delay_m)]))
    Y = np.vstack(stacked_rows)
    mean_var2 = float(np.var(Y, axis=0).mean())
    if mean_var2 < var_threshold:
        raise DegenerateEmbeddingError(
            f"Embeddings collapsed even after time-delay (mean_var={mean_var2:.3e})"
        )
    norms = np.linalg.norm(Y, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Y = Y / (norms + 1e-12)
    return Y
