from typing import Dict

import numpy as np


class WitnessValidationError(RuntimeError):
    """Raised when witness matrices violate numerical preconditions."""


def compute_spectral_gap(svals: np.ndarray) -> float:
    """Compute the gap between the top two singular values."""
    if len(svals) < 2:
        return float("inf")
    return float(svals[0] - svals[1])


def check_witness(
    X0: np.ndarray,
    X1: np.ndarray,
    A: np.ndarray,
    k: int,
    cond_thresh: float = 1e4,
    gap_thresh: float = 1e-12,
) -> Dict[str, float]:
    """Validate numerical preconditions for the verified kernel.

    Raises WitnessValidationError on failure. Returns diagnostics on success.
    """
    if not (isinstance(X0, np.ndarray) and isinstance(X1, np.ndarray) and isinstance(A, np.ndarray)):
        raise WitnessValidationError("Witness inputs must be numpy arrays")

    if not np.isfinite(X0).all():
        raise WitnessValidationError("X0 contains non-finite values")
    if not np.isfinite(X1).all():
        raise WitnessValidationError("X1 contains non-finite values")
    if not np.isfinite(A).all():
        raise WitnessValidationError("A contains non-finite values")

    if X0.ndim != 2 or X1.ndim != 2:
        raise WitnessValidationError("X0/X1 must be 2D")
    if X0.shape[1] != X1.shape[1]:
        raise WitnessValidationError("X0 and X1 must have same number of columns (timesteps)")

    n_cols = X0.shape[1]

    gram = X0 @ X0.T
    try:
        cond = np.linalg.cond(gram)
    except Exception:
        cond = float("inf")
    if cond > cond_thresh:
        raise WitnessValidationError(
            f"Condition number too large: {cond:.3e} > {cond_thresh:.3e}"
        )

    try:
        svals = np.linalg.svd(X0, compute_uv=False)
    except Exception:
        svals = np.array([0.0])
    spectral_gap = compute_spectral_gap(svals)
    if spectral_gap < gap_thresh:
        raise WitnessValidationError(
            f"Singular value gap too small: {spectral_gap:.3e} < {gap_thresh:.3e}"
        )

    if not (1 <= k <= min(X0.shape[0], X0.shape[1])):
        raise WitnessValidationError(f"k ({k}) inconsistent with shapes {X0.shape}")

    return {
        "condition_number": float(cond),
        "spectral_gap": float(spectral_gap),
        "r_eff_checked": int(k),
        "n_train_cols": int(n_cols),
    }


def check_witness_properties(
    X0: np.ndarray,
    X1: np.ndarray,
    A: np.ndarray,
    cond_thresh: float = 1e4,
    gap_thresh: float = 1e-12,
) -> Dict[str, float]:
    """Backward-compatible witness validation wrapper for kernel bridge."""
    r_eff = min(X0.shape[0], X0.shape[1]) if isinstance(X0, np.ndarray) and X0.ndim == 2 else 1
    return check_witness(
        X0,
        X1,
        A,
        k=r_eff,
        cond_thresh=cond_thresh,
        gap_thresh=gap_thresh,
    )
