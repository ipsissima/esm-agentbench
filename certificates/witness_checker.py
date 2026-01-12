"""Witness validation for spectral certificate kernel inputs.

This module provides strict checks to ensure numerical preconditions
for verified kernel calls are satisfied.
"""
from __future__ import annotations

import numpy as np


class WitnessValidationError(ValueError):
    """Raised when witness matrices fail pre-kernel validation."""


def _require_finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise WitnessValidationError(f"{name} contains NaN or Inf values")


def _require_matrix(name: str, value: np.ndarray) -> None:
    if not isinstance(value, np.ndarray):
        raise WitnessValidationError(f"{name} must be a numpy array, got {type(value)}")
    if value.ndim != 2:
        raise WitnessValidationError(f"{name} must be 2D, got shape {value.shape}")


def check_witness_properties(
    X0: np.ndarray,
    X1: np.ndarray,
    A: np.ndarray,
    *,
    min_singular_gap: float = 1e-5,
) -> None:
    """Validate witness matrices before kernel evaluation.

    Parameters
    ----------
    X0, X1, A : np.ndarray
        Witness matrices passed to the verified kernel.
    min_singular_gap : float
        Minimum acceptable singular gap between sigma_k and sigma_{k+1}.

    Raises
    ------
    WitnessValidationError
        If any numerical preconditions fail.
    """
    _require_matrix("X0", X0)
    _require_matrix("X1", X1)
    _require_matrix("A", A)

    _require_finite("X0", X0)
    _require_finite("X1", X1)
    _require_finite("A", A)

    if X0.shape != X1.shape:
        raise WitnessValidationError(
            f"X0 and X1 must share the same shape, got {X0.shape} vs {X1.shape}"
        )

    if A.shape[0] != A.shape[1]:
        raise WitnessValidationError(f"A must be square, got shape {A.shape}")

    if A.shape[0] != X0.shape[0]:
        raise WitnessValidationError(
            "A must align with witness rank: "
            f"A is {A.shape}, X0 is {X0.shape}"
        )

    k_eff = A.shape[0]
    if k_eff < 1:
        raise WitnessValidationError(f"A must be non-empty, got shape {A.shape}")

    if k_eff > min(X0.shape[0], X0.shape[1]):
        raise WitnessValidationError(
            f"A dimension k={k_eff} exceeds witness dimensions {X0.shape}"
        )

    # Rank verification: A should be full rank relative to its dimensions.
    rank_A = np.linalg.matrix_rank(A)
    if rank_A != k_eff:
        raise WitnessValidationError(
            "Effective rank mismatch: "
            f"rank(A)={rank_A} does not match A dimension k={k_eff}"
        )

    # Condition number check
    cond_threshold = 1e4
    cond_A = float(np.linalg.cond(A))
    if not np.isfinite(cond_A):
        raise WitnessValidationError("Condition number of A is not finite")
    if cond_A > cond_threshold:
        raise WitnessValidationError(
            f"Condition number of A too large: cond(A)={cond_A:.3e}"
        )

    # Singular gap check between sigma_k and sigma_{k+1}
    singular_values = np.linalg.svd(A, compute_uv=False)
    if k_eff > len(singular_values):
        raise WitnessValidationError(
            f"k={k_eff} exceeds available singular values ({len(singular_values)})"
        )

    sigma_k = float(singular_values[k_eff - 1])
    sigma_k_plus_1 = float(singular_values[k_eff]) if k_eff < len(singular_values) else 0.0
    singular_gap = sigma_k - sigma_k_plus_1

    if singular_gap <= min_singular_gap:
        raise WitnessValidationError(
            "Singular gap too small: "
            f"sigma_k={sigma_k:.3e}, sigma_k+1={sigma_k_plus_1:.3e}, "
            f"gap={singular_gap:.3e}, min_gap={min_singular_gap:.3e}"
        )
