"""Witness validation for spectral certificate kernel inputs.

This module provides strict checks to ensure numerical preconditions
for Wedin's Theorem and verified kernel calls are satisfied.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


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


def check_witness(
    X0: np.ndarray,
    X1: np.ndarray,
    A: np.ndarray,
    *,
    k: Optional[int] = None,
    cond_threshold: float = 1e4,
    gap_threshold: Optional[float] = None,
    strict: bool = True,
) -> None:
    """Validate witness matrices before kernel evaluation.

    Parameters
    ----------
    X0, X1, A : np.ndarray
        Witness matrices passed to the verified kernel.
    k : Optional[int]
        Rank used for truncation; defaults to X0.shape[0].
    cond_threshold : float
        Maximum acceptable condition number for A.
    gap_threshold : Optional[float]
        Minimum acceptable singular gap; defaults to machine epsilon scale.
    strict : bool
        If True, raise on violations. If False, log warnings for gap issues.

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

    k_eff = k if k is not None else X0.shape[0]
    if k_eff < 1:
        raise WitnessValidationError(f"k must be positive, got {k_eff}")

    if k_eff > min(X0.shape[0], X0.shape[1]):
        raise WitnessValidationError(
            f"k={k_eff} exceeds witness dimensions {X0.shape}"
        )

    # Dimensionality check: T > 2k to avoid overfitting
    T = X0.shape[1] + 1
    if T <= 2 * k_eff:
        raise WitnessValidationError(
            f"Insufficient timesteps: T={T} must exceed 2k={2 * k_eff}"
        )

    # Rank verification
    rank_X0 = np.linalg.matrix_rank(X0)
    if k_eff > rank_X0:
        raise WitnessValidationError(
            f"Effective rank k={k_eff} exceeds numerical rank of X0 ({rank_X0})"
        )

    # Condition number check
    cond_A = float(np.linalg.cond(A))
    if not np.isfinite(cond_A):
        raise WitnessValidationError("Condition number of A is not finite")
    if cond_A > cond_threshold:
        raise WitnessValidationError(
            f"Condition number of A too large: cond(A)={cond_A:.3e}"
        )

    # Singular gap check for Wedin's theorem
    singular_values = np.linalg.svd(A, compute_uv=False)
    if k_eff > len(singular_values):
        raise WitnessValidationError(
            f"k={k_eff} exceeds available singular values ({len(singular_values)})"
        )

    sigma_k = float(singular_values[k_eff - 1])
    sigma_k_plus_1 = float(singular_values[k_eff]) if k_eff < len(singular_values) else 0.0
    singular_gap = sigma_k - sigma_k_plus_1

    eps = np.finfo(singular_values.dtype).eps
    gap_floor = gap_threshold if gap_threshold is not None else eps * max(1.0, sigma_k)

    if singular_gap <= gap_floor:
        message = (
            "Singular gap too small for Wedin's theorem: "
            f"sigma_k={sigma_k:.3e}, sigma_k+1={sigma_k_plus_1:.3e}, "
            f"gap={singular_gap:.3e}"
        )
        if strict:
            raise WitnessValidationError(message)
        logger.warning(message)
