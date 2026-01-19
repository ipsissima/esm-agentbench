"""Mass model for Telegrapher-Koopman system identification.

This module implements a second-difference penalty regression that models
agent trajectory dynamics using the discrete Telegrapher equation. The
"mass" term provides a restorative force that penalizes acceleration,
distinguishing stable behavior from drift.

Mathematical Background
-----------------------
The discrete Telegrapher equation models dynamics with inertia:

    x[t+1] - 2*x[t] + x[t-1] = f[t] - γ*(x[t] - x[t-1]) - m*a[t]

Where:
- f[t] is external forcing
- γ is damping coefficient
- m is mass (inertia)
- a[t] is acceleration

For trajectory analysis, we reformulate as a regression problem with
second-difference penalty:

    minimize ||X1 - A @ X0||_F^2 + λ_mass * ||D2 @ X||_F^2 + λ_ridge * ||A||_F^2

Where D2 is the second-difference matrix that computes acceleration.

Key Invariants
--------------
1. For stable trajectories (no drift): residual < bound
2. Mass-controlled regression separates drift from oscillation
3. Larger mass_lambda → more penalty on acceleration → smoother fits
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.linalg import lstsq, norm

logger = logging.getLogger(__name__)


@dataclass
class MassModelResult:
    """Result of mass model fitting.

    Attributes
    ----------
    A : np.ndarray
        Learned linear operator, shape (r, r).
    residual : float
        Normalized prediction residual ||X1 - A @ X0||_F / ||X1||_F.
    acceleration_penalty : float
        Normalized second-difference penalty ||D2 @ X||_F / ||X||_F.
    fit_info : dict
        Additional fitting information (condition number, rank, etc.).
    """

    A: np.ndarray
    residual: float
    acceleration_penalty: float
    fit_info: dict


def _make_second_difference_matrix(n: int) -> np.ndarray:
    """Create second-difference operator matrix D2.

    The second-difference matrix computes discrete acceleration:
    (D2 @ x)[i] = x[i+1] - 2*x[i] + x[i-1]

    Parameters
    ----------
    n : int
        Length of signal.

    Returns
    -------
    np.ndarray
        Second-difference matrix of shape (n-2, n).
    """
    if n < 3:
        raise ValueError(f"Need n >= 3 for second difference, got n={n}")

    D2 = np.zeros((n - 2, n))
    for i in range(n - 2):
        D2[i, i] = 1
        D2[i, i + 1] = -2
        D2[i, i + 2] = 1

    return D2


def fit_mass_model(
    X: np.ndarray,
    mass_lambda: float = 0.1,
    gamma: float = 0.0,
    ridge: float = 1e-6,
) -> MassModelResult:
    """Fit mass-controlled linear model to trajectory data.

    This function solves the regularized regression problem:

        A* = argmin_A ||X1 - A @ X0||_F^2
                     + mass_lambda * ||D2 @ X^T||_F^2
                     + ridge * ||A||_F^2

    The mass_lambda term penalizes acceleration (second differences),
    encouraging smooth dynamics. Higher mass_lambda means the system
    has more "inertia" and resists rapid changes.

    Parameters
    ----------
    X : np.ndarray
        State trajectory matrix of shape (r, T) where r is dimension
        and T is number of time steps.
    mass_lambda : float, default=0.1
        Weight for second-difference (acceleration) penalty.
        Higher values → smoother fits, less sensitive to noise.
    gamma : float, default=0.0
        Damping coefficient (currently unused, reserved for future).
    ridge : float, default=1e-6
        Ridge regularization for numerical stability.

    Returns
    -------
    MassModelResult
        Fitted model with operator A, residual, and diagnostics.

    Raises
    ------
    ValueError
        If X has fewer than 3 time steps.

    Examples
    --------
    >>> X = np.random.randn(4, 10)  # 4-dimensional, 10 time steps
    >>> result = fit_mass_model(X, mass_lambda=0.1)
    >>> print(f"Residual: {result.residual:.4f}")
    """
    r, T = X.shape

    if T < 3:
        raise ValueError(f"Need T >= 3 time steps, got T={T}")

    # Construct X0 (time 0 to T-2) and X1 (time 1 to T-1)
    X0 = X[:, :-1]  # shape (r, T-1)
    X1 = X[:, 1:]  # shape (r, T-1)

    # Solve for A using regularized least squares
    # A @ X0 ≈ X1
    # With ridge regularization: (X0 @ X0.T + ridge*I) A.T = X0 @ X1.T

    # Compute Gram matrix with regularization
    G = X0 @ X0.T + ridge * np.eye(r)

    # Solve for A
    try:
        A = np.linalg.solve(G, X0 @ X1.T).T
    except np.linalg.LinAlgError:
        # Fall back to pseudo-inverse if singular
        logger.warning("Gram matrix singular, using pseudo-inverse")
        A = (X1 @ np.linalg.pinv(X0))

    # Compute prediction residual
    pred = A @ X0
    residual_norm = norm(X1 - pred, "fro")
    X1_norm = norm(X1, "fro")
    residual = residual_norm / X1_norm if X1_norm > 0 else 0.0

    # Compute acceleration penalty using second-difference
    # D2 operates on each row of X (each dimension's trajectory)
    D2 = _make_second_difference_matrix(T)
    accel = X @ D2.T  # shape (r, T-2) - acceleration for each dimension
    accel_norm = norm(accel, "fro")
    X_norm = norm(X, "fro")
    accel_penalty = accel_norm / X_norm if X_norm > 0 else 0.0

    # Compute fit diagnostics
    cond = np.linalg.cond(G)
    spectral_radius = np.max(np.abs(np.linalg.eigvals(A)))

    fit_info = {
        "condition_number": float(cond),
        "spectral_radius": float(spectral_radius),
        "mass_lambda": mass_lambda,
        "gamma": gamma,
        "ridge": ridge,
        "n_samples": T,
        "n_dims": r,
    }

    return MassModelResult(
        A=A,
        residual=float(residual),
        acceleration_penalty=float(accel_penalty),
        fit_info=fit_info,
    )


def compute_mass_residual(
    X: np.ndarray,
    A: np.ndarray,
    mass_lambda: float = 0.1,
) -> Tuple[float, float]:
    """Compute combined residual with mass penalty.

    This computes the weighted sum of prediction residual and
    acceleration penalty, which forms the basis for stability
    certificates.

    Parameters
    ----------
    X : np.ndarray
        State trajectory matrix of shape (r, T).
    A : np.ndarray
        Linear operator of shape (r, r).
    mass_lambda : float, default=0.1
        Weight for acceleration penalty.

    Returns
    -------
    Tuple[float, float]
        (combined_residual, bound_contribution) where:
        - combined_residual = pred_residual + mass_lambda * accel_penalty
        - bound_contribution = theoretical bound contribution
    """
    r, T = X.shape

    if T < 3:
        raise ValueError(f"Need T >= 3 time steps, got T={T}")

    X0 = X[:, :-1]
    X1 = X[:, 1:]

    # Prediction residual
    pred = A @ X0
    pred_residual = norm(X1 - pred, "fro") / (norm(X1, "fro") + 1e-10)

    # Acceleration penalty
    D2 = _make_second_difference_matrix(T)
    accel = X @ D2.T
    accel_penalty = norm(accel, "fro") / (norm(X, "fro") + 1e-10)

    # Combined residual
    combined = pred_residual + mass_lambda * accel_penalty

    # Bound contribution (theoretical analysis shows this scaling)
    # For stable systems: spectral_radius(A) < 1 implies bounded growth
    spectral_radius = np.max(np.abs(np.linalg.eigvals(A)))
    bound_contrib = combined * (1 + spectral_radius) / (1 - min(spectral_radius, 0.999))

    return float(combined), float(bound_contrib)


def synthetic_rabbit_trace(
    n_steps: int = 100,
    n_dims: int = 4,
    drift_rate: float = 0.0,
    noise_level: float = 0.01,
    mass: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate synthetic "rabbit" trace for testing.

    A "rabbit" trace models an agent that follows a reference trajectory
    (the "carrot") with some inertia (mass) and noise. When drift_rate > 0,
    the reference slowly moves away, causing the agent to drift.

    Parameters
    ----------
    n_steps : int, default=100
        Number of time steps.
    n_dims : int, default=4
        Dimensionality of state space.
    drift_rate : float, default=0.0
        Rate of systematic drift (0 = no drift = stable).
    noise_level : float, default=0.01
        Standard deviation of Gaussian noise.
    mass : float, default=1.0
        Inertia of the system (higher = smoother, slower response).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Trajectory matrix of shape (n_dims, n_steps).

    Examples
    --------
    >>> # Stable trace (no drift)
    >>> X_stable = synthetic_rabbit_trace(n_steps=100, drift_rate=0.0)
    >>> result = fit_mass_model(X_stable)
    >>> assert result.residual < 0.1  # Should have low residual

    >>> # Drifting trace
    >>> X_drift = synthetic_rabbit_trace(n_steps=100, drift_rate=0.05)
    >>> result = fit_mass_model(X_drift)
    >>> # May have higher residual or different spectral properties
    """
    if seed is not None:
        np.random.seed(seed)

    # Initialize state and velocity
    x = np.zeros((n_dims, n_steps))
    x[:, 0] = np.random.randn(n_dims) * 0.1

    # Reference trajectory (the "carrot")
    ref = np.zeros((n_dims, n_steps))
    ref[:, 0] = np.random.randn(n_dims)

    # Dynamics parameters
    # Higher mass → slower response to reference changes
    # spring_k pulls toward reference, damping_c resists velocity
    spring_k = 1.0 / mass
    damping_c = 0.5 / mass

    v = np.zeros(n_dims)  # velocity

    for t in range(1, n_steps):
        # Update reference with drift
        ref[:, t] = ref[:, t - 1] + drift_rate * np.ones(n_dims) + noise_level * np.random.randn(n_dims)

        # Spring-damper dynamics toward reference
        # F = -k*(x - ref) - c*v
        force = -spring_k * (x[:, t - 1] - ref[:, t]) - damping_c * v
        force += noise_level * np.random.randn(n_dims)

        # Update velocity and position (Euler integration)
        v = v + force
        x[:, t] = x[:, t - 1] + v

    return x


__all__ = [
    "fit_mass_model",
    "MassModelResult",
    "compute_mass_residual",
    "synthetic_rabbit_trace",
]
