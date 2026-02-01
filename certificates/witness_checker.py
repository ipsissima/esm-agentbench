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
    relative_gap_thresh: float = 1e-6,
) -> Dict[str, float]:
    """Validate numerical preconditions for the verified kernel.

    The spectral gap check now includes both absolute and relative thresholds
    to prevent false positives when singular values are very small.

    Parameters
    ----------
    X0 : np.ndarray
        State matrix at time t, shape (r, n).
    X1 : np.ndarray
        State matrix at time t+1, shape (r, n).
    A : np.ndarray
        Learned operator, shape (r, r).
    k : int
        Effective rank to check.
    cond_thresh : float
        Maximum allowed condition number (default: 1e4).
    gap_thresh : float
        Minimum absolute spectral gap (default: 1e-12).
    relative_gap_thresh : float
        Minimum relative spectral gap: gap / sigma_max (default: 1e-6).
        This prevents false positives when all singular values are very small.

    Returns
    -------
    Dict[str, float]
        Diagnostics including condition number, absolute and relative gaps.

    Raises
    ------
    WitnessValidationError
        If witness matrices violate numerical preconditions.
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

    try:
        svals = np.linalg.svd(X0, compute_uv=False)
    except Exception:
        svals = np.array([0.0])

    if len(svals) == 0 or svals[-1] == 0:
        cond = float("inf")
    else:
        cond = float(svals[0] / svals[-1])
    if cond > cond_thresh:
        raise WitnessValidationError(
            f"Condition number too large: {cond:.3e} > {cond_thresh:.3e}"
        )
    
    # Compute spectral gap (absolute and relative)
    spectral_gap = compute_spectral_gap(svals)
    
    # Compute relative gap to handle small singular values
    if len(svals) > 0 and svals[0] > 0:
        relative_gap = spectral_gap / svals[0]
    else:
        relative_gap = 0.0
    
    # Check both absolute and relative gap thresholds
    # The relative gap check prevents false positives when all singular values are tiny
    if spectral_gap < gap_thresh and relative_gap < relative_gap_thresh:
        raise WitnessValidationError(
            f"Singular value gap too small: absolute={spectral_gap:.3e} < {gap_thresh:.3e}, "
            f"relative={relative_gap:.3e} < {relative_gap_thresh:.3e}"
        )

    if not (1 <= k <= min(X0.shape[0], X0.shape[1])):
        raise WitnessValidationError(f"k ({k}) inconsistent with shapes {X0.shape}")

    return {
        "condition_number": float(cond),
        "spectral_gap": float(spectral_gap),
        "relative_gap": float(relative_gap),
        "r_eff_checked": int(k),
        "n_train_cols": int(n_cols),
    }


def check_witness_properties(
    X0: np.ndarray,
    X1: np.ndarray,
    A: np.ndarray,
    cond_thresh: float = 1e4,
    gap_thresh: float = 1e-12,
    relative_gap_thresh: float = 1e-6,
) -> Dict[str, float]:
    """Backward-compatible witness validation wrapper for kernel bridge.
    
    Parameters
    ----------
    X0, X1, A : np.ndarray
        Witness matrices.
    cond_thresh : float
        Maximum allowed condition number.
    gap_thresh : float
        Minimum absolute spectral gap.
    relative_gap_thresh : float
        Minimum relative spectral gap.
    
    Returns
    -------
    Dict[str, float]
        Diagnostics from witness validation.
    """
    r_eff = min(X0.shape[0], X0.shape[1]) if isinstance(X0, np.ndarray) and X0.ndim == 2 else 1
    return check_witness(
        X0,
        X1,
        A,
        k=r_eff,
        cond_thresh=cond_thresh,
        gap_thresh=gap_thresh,
        relative_gap_thresh=relative_gap_thresh,
    )
