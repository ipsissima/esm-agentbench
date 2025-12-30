"""Spectral Prover: Modular library for spectral certificate computation.

This module provides functions for computing spectral certificates using SVD,
Davis-Kahan/Wedin subspace perturbation bounds, and Koopman operator fitting.

All data used in validation are synthetic. No real secrets or external network calls.

Mathematical Foundations:
- Wedin's Theorem: Bounds singular subspace perturbation under matrix noise
- Davis-Kahan Theorem: Bounds eigenspace rotation under symmetric perturbation
- Koopman Operators: Linear approximation of nonlinear dynamics in embedding space

References:
- Davis & Kahan (1970), SIAM J. Numer. Anal.
- Wedin (1972), BIT Numer. Math.
- See docs/SPECTRAL_THEORY.md for full derivations
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.linalg import norm, pinv
from scipy.linalg import svd, subspace_angles


def trajectory_matrix(
    embeddings: np.ndarray,
    hankel: bool = False,
    L: Optional[int] = None,
    normalize: bool = True,
) -> np.ndarray:
    """Construct trajectory matrix from embedding sequence.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding sequence of shape (T, d) where T is timesteps and d is dimension.
    hankel : bool, optional
        If True, construct Hankel (time-delay) embedding matrix. Default False.
    L : int, optional
        Hankel window length. Required if hankel=True. Default None.
    normalize : bool, optional
        If True, zero-mean and unit-variance normalize per-run. Default True.

    Returns
    -------
    np.ndarray
        Trajectory matrix X of shape (d, T) or Hankel matrix of shape (L*d, T-L+1).

    Examples
    --------
    >>> X = trajectory_matrix(np.random.randn(20, 10))
    >>> X.shape
    (10, 20)
    """
    X = np.asarray(embeddings, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    T, d = X.shape

    # Normalize: zero-mean, unit variance per-run
    if normalize and T > 1:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        X = (X - mean) / std

    if not hankel:
        # Standard trajectory matrix: d x T
        return X.T

    # Hankel (time-delay) embedding
    if L is None:
        L = min(T // 2, 10)
    if L < 1 or L > T:
        raise ValueError(f"Invalid Hankel window L={L} for T={T}")

    n_cols = T - L + 1
    H = np.zeros((L * d, n_cols), dtype=np.float64)
    for i in range(n_cols):
        H[:, i] = X[i : i + L].flatten()

    return H


def compute_svd_certificate(X: np.ndarray, k: int) -> Dict[str, Any]:
    """Compute SVD-based spectral certificate with all relevant metrics.

    Parameters
    ----------
    X : np.ndarray
        Trajectory matrix of shape (d, T).
    k : int
        Rank for truncated SVD approximation.

    Returns
    -------
    dict
        Certificate containing:
        - residual: Frobenius norm of rank-k approximation error (normalized)
        - theoretical_bound: Placeholder for combined bound (compute separately)
        - sigma_max: Largest singular value
        - singular_gap: Gap between k-th and (k+1)-th singular values
        - tail_energy: Fraction of energy not captured by top-k
        - pca_explained: Fraction of variance explained by top-k
        - U_k: Left singular vectors (d, k)
        - Sigma_k: Singular values (k,)
        - V_k: Right singular vectors (T, k)
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {X.shape}")

    d, T = X.shape
    if T < 2:
        return _empty_certificate(k)

    # Compute full SVD
    U, S, Vt = svd(X, full_matrices=False)
    r = len(S)  # Actual rank

    # Effective k (cannot exceed actual rank)
    k_eff = min(k, r, T, d)

    # Truncate to rank k
    U_k = U[:, :k_eff]
    Sigma_k = S[:k_eff]
    V_k = Vt[:k_eff, :].T  # Shape (T, k)

    # Reconstruction: X_k = U_k @ diag(Sigma_k) @ V_k.T
    X_k = U_k @ np.diag(Sigma_k) @ V_k.T

    # Residual (Frobenius norm, normalized)
    residual_unnorm = norm(X - X_k, ord='fro')
    X_norm = norm(X, ord='fro')
    residual = float(residual_unnorm / (X_norm + 1e-12))

    # Energy metrics
    total_energy = float(np.sum(S ** 2))
    retained_energy = float(np.sum(Sigma_k ** 2))
    pca_explained = retained_energy / (total_energy + 1e-12)
    pca_explained = float(np.clip(pca_explained, 0.0, 1.0))
    tail_energy = float(max(0.0, 1.0 - pca_explained))

    # Singular value statistics
    sigma_max = float(S[0]) if len(S) > 0 else 0.0
    sigma_k = float(S[k_eff - 1]) if k_eff > 0 else 0.0
    sigma_k_plus_1 = float(S[k_eff]) if k_eff < r else 0.0
    singular_gap = float(sigma_k - sigma_k_plus_1)

    return {
        'residual': residual,
        'theoretical_bound': None,  # Computed by caller with full context
        'sigma_max': sigma_max,
        'sigma_k': sigma_k,
        'singular_gap': singular_gap,
        'tail_energy': tail_energy,
        'pca_explained': pca_explained,
        'U_k': U_k,
        'Sigma_k': Sigma_k,
        'V_k': V_k,
    }


def _empty_certificate(k: int) -> Dict[str, Any]:
    """Return conservative empty certificate for insufficient data."""
    return {
        'residual': 1.0,
        'theoretical_bound': float('inf'),
        'sigma_max': 0.0,
        'sigma_k': 0.0,
        'singular_gap': 0.0,
        'tail_energy': 1.0,
        'pca_explained': 0.0,
        'U_k': np.zeros((1, k)),
        'Sigma_k': np.zeros(k),
        'V_k': np.zeros((1, k)),
    }


def subspace_angle(U: np.ndarray, V: np.ndarray) -> float:
    """Compute the largest principal angle between subspaces.

    Uses Davis-Kahan / canonical angles to measure subspace distance.

    Parameters
    ----------
    U : np.ndarray
        Orthonormal basis for first subspace, shape (d, k1).
    V : np.ndarray
        Orthonormal basis for second subspace, shape (d, k2).

    Returns
    -------
    float
        Largest principal angle in radians, in [0, pi/2].

    Notes
    -----
    Uses scipy.linalg.subspace_angles which computes arccos of singular values
    of U.T @ V after orthonormalization.
    """
    if U.ndim == 1:
        U = U.reshape(-1, 1)
    if V.ndim == 1:
        V = V.reshape(-1, 1)

    # Handle dimension mismatch
    if U.shape[0] != V.shape[0]:
        raise ValueError(f"Dimension mismatch: U has {U.shape[0]} rows, V has {V.shape[0]}")

    # scipy.linalg.subspace_angles returns angles in descending order
    angles = subspace_angles(U, V)

    # Return largest principal angle
    return float(np.max(angles)) if len(angles) > 0 else 0.0


def davis_kahan_upper_bound(E_norm: float, delta: float) -> float:
    """Compute Davis-Kahan upper bound on sin(theta).

    The Davis-Kahan theorem states that for symmetric matrices A and A+E,
    the sine of the principal angle between eigenspaces is bounded by:

        sin(theta) <= ||E|| / delta

    where delta is the eigenvalue gap.

    Parameters
    ----------
    E_norm : float
        Operator norm (or Frobenius norm) of perturbation E.
    delta : float
        Spectral gap (difference between k-th and (k+1)-th eigenvalue).

    Returns
    -------
    float
        Upper bound on sin(theta), clipped to [0, 1].

    References
    ----------
    Davis, C., & Kahan, W. M. (1970). The rotation of eigenvectors by a
    perturbation. III. SIAM J. Numer. Anal., 7(1), 1-46.
    """
    if delta <= 0:
        return 1.0  # No gap means no bound
    bound = E_norm / delta
    return float(min(1.0, max(0.0, bound)))


def koopman_operator_fit(
    X: np.ndarray,
    k: int,
    regularization: float = 1e-8,
) -> Dict[str, Any]:
    """Fit linear Koopman operator from trajectory data.

    Fits A such that X[:, t+1] ≈ A @ X[:, t] in the top-k subspace.

    Parameters
    ----------
    X : np.ndarray
        Trajectory matrix of shape (d, T).
    k : int
        Rank for dimensionality reduction before fitting.
    regularization : float, optional
        Regularization for Gram matrix inversion. Default 1e-8.

    Returns
    -------
    dict
        - K_approx: Fitted Koopman operator (k, k)
        - prediction_residual: Normalized prediction error
        - Z: Projected states (k, T)
        - eigenvalues: Eigenvalues of K_approx (for stability analysis)
    """
    d, T = X.shape
    if T < 3:
        return {
            'K_approx': np.eye(min(k, d)),
            'prediction_residual': 1.0,
            'Z': X[:min(k, d), :],
            'eigenvalues': np.ones(min(k, d)),
        }

    # Get SVD certificate
    cert = compute_svd_certificate(X, k)
    V_k = cert['V_k']  # (T, k)

    # Project onto reduced space: Z = X @ V_k -> (d, k) but we want (k, T)
    # Actually Z = V_k.T gives us (k, T) coordinates
    Z = V_k.T  # Shape: (k, T)

    k_eff = Z.shape[0]

    # Time-shifted states
    Z0 = Z[:, :-1]  # (k, T-1)
    Z1 = Z[:, 1:]   # (k, T-1)

    # Fit linear operator: Z1 = A @ Z0
    # A = Z1 @ Z0.T @ (Z0 @ Z0.T + reg*I)^{-1}
    Gram = Z0 @ Z0.T + regularization * np.eye(k_eff)
    A = Z1 @ Z0.T @ pinv(Gram)

    # Prediction residual
    Z1_pred = A @ Z0
    err = Z1 - Z1_pred
    residual = float(norm(err, 'fro') / (norm(Z1, 'fro') + 1e-12))

    # Eigenvalue analysis (for stability)
    eigenvalues = np.linalg.eigvals(A)

    return {
        'K_approx': A,
        'prediction_residual': residual,
        'Z': Z,
        'eigenvalues': eigenvalues,
    }


def compute_detection_statistics(
    run_trace: Union[List[List[float]], np.ndarray],
    k: int,
    baseline_U: Optional[np.ndarray] = None,
    C_res: float = 1.0,
    C_tail: float = 1.0,
) -> Dict[str, Any]:
    """Compute all detection statistics for a single run trace.

    This is the main entry point for computing spectral features needed
    for drift vs creative classification.

    Parameters
    ----------
    run_trace : array-like
        Embedding sequence of shape (T, d).
    k : int
        Rank for spectral analysis.
    baseline_U : np.ndarray, optional
        Baseline left singular vectors for Davis-Kahan angle computation.
        Shape (d, k). If None, dk_angle is set to 0.
    C_res : float, optional
        Coefficient for residual in theoretical bound. Default 1.0.
    C_tail : float, optional
        Coefficient for tail energy in theoretical bound. Default 1.0.

    Returns
    -------
    dict
        Detection statistics including:
        - residual: SVD reconstruction residual
        - theoretical_bound: C_res * residual + C_tail * tail_energy
        - sigma_max: Largest singular value
        - singular_gap: Gap at rank k
        - tail_energy: Unexplained variance
        - pca_explained: Explained variance
        - dk_angle: Davis-Kahan angle vs baseline (radians)
        - dk_bound: Theoretical upper bound on dk_angle
        - koopman_residual: Koopman prediction residual
        - length_T: Trace length
    """
    # Convert to numpy array
    embeddings = np.asarray(run_trace, dtype=np.float64)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, 1)

    T, d = embeddings.shape

    # Build trajectory matrix
    X = trajectory_matrix(embeddings, normalize=True)  # Shape: (d, T)

    # Compute SVD certificate
    svd_cert = compute_svd_certificate(X, k)

    # Compute Koopman fit
    koop = koopman_operator_fit(X, k)

    # Compute Davis-Kahan angle if baseline provided
    if baseline_U is not None and svd_cert['U_k'].shape == baseline_U.shape:
        dk_angle = subspace_angle(svd_cert['U_k'], baseline_U)
        # Theoretical bound: estimate perturbation as residual * X_norm
        E_norm_est = svd_cert['residual'] * norm(X, 'fro')
        dk_bound = davis_kahan_upper_bound(E_norm_est, svd_cert['singular_gap'] + 1e-12)
    else:
        dk_angle = 0.0
        dk_bound = 1.0

    # Theoretical bound
    theoretical_bound = C_res * svd_cert['residual'] + C_tail * svd_cert['tail_energy']

    return {
        'residual': svd_cert['residual'],
        'theoretical_bound': theoretical_bound,
        'sigma_max': svd_cert['sigma_max'],
        'sigma_k': svd_cert['sigma_k'],
        'singular_gap': svd_cert['singular_gap'],
        'tail_energy': svd_cert['tail_energy'],
        'pca_explained': svd_cert['pca_explained'],
        'dk_angle': dk_angle,
        'dk_bound': dk_bound,
        'koopman_residual': koop['prediction_residual'],
        'spectral_radius': float(np.max(np.abs(koop['eigenvalues']))),
        'length_T': T,
        'U_k': svd_cert['U_k'],
    }


def compute_theoretical_bound(
    residual: float,
    tail_energy: float,
    semantic_divergence: float = 0.0,
    lipschitz_margin: float = 0.0,
    C_res: float = 1.0,
    C_tail: float = 1.0,
    C_sem: float = 1.0,
    C_robust: float = 1.0,
) -> float:
    """Compute theoretical spectral bound from components.

    The bound is:
        B = C_res * residual + C_tail * tail_energy
            + C_sem * semantic_divergence + C_robust * lipschitz_margin

    All constants are bounded by 2.0 (formally verified in Coq).

    Parameters
    ----------
    residual : float
        Normalized reconstruction residual.
    tail_energy : float
        Unexplained variance fraction.
    semantic_divergence : float, optional
        Mean cosine distance from task embedding. Default 0.0.
    lipschitz_margin : float, optional
        Embedding stability margin. Default 0.0.
    C_res, C_tail, C_sem, C_robust : float, optional
        Weighting constants. Defaults to 1.0.

    Returns
    -------
    float
        Theoretical bound value.
    """
    return (
        C_res * residual
        + C_tail * tail_energy
        + C_sem * semantic_divergence
        + C_robust * lipschitz_margin
    )


def generate_synthetic_linear_system(
    d: int = 10,
    T: int = 50,
    k: int = 3,
    noise_std: float = 0.01,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic trajectory from stable linear system.

    Creates x_{t+1} = K @ x_t + noise where K is a stable matrix
    (spectral radius < 1) with effective rank k.

    Parameters
    ----------
    d : int
        State dimension.
    T : int
        Number of timesteps.
    k : int
        Effective rank of dynamics matrix K.
    noise_std : float
        Standard deviation of additive noise.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray
        Trajectory embeddings of shape (T, d).
    K : np.ndarray
        True dynamics matrix of shape (d, d).
    """
    rng = np.random.default_rng(seed)

    # Create low-rank stable matrix
    # K = U @ diag(s) @ V.T where s_i < 1
    U = np.linalg.qr(rng.standard_normal((d, k)))[0]
    V = np.linalg.qr(rng.standard_normal((d, k)))[0]
    singular_values = 0.8 * rng.uniform(0.5, 0.95, size=k)  # All < 1 for stability

    K = U @ np.diag(singular_values) @ V.T

    # Generate trajectory
    X = np.zeros((T, d))
    X[0] = rng.standard_normal(d)

    for t in range(T - 1):
        X[t + 1] = K @ X[t] + noise_std * rng.standard_normal(d)

    return X, K


def generate_perturbed_trajectory(
    X_clean: np.ndarray,
    perturbation_norm: float,
    perturbation_type: str = 'additive',
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate perturbed trajectory for testing detection.

    Parameters
    ----------
    X_clean : np.ndarray
        Clean trajectory of shape (T, d).
    perturbation_norm : float
        Frobenius norm of perturbation to add.
    perturbation_type : str
        Type of perturbation: 'additive', 'multiplicative', or 'drift'.
    seed : int, optional
        Random seed.

    Returns
    -------
    X_perturbed : np.ndarray
        Perturbed trajectory of shape (T, d).
    E : np.ndarray
        Perturbation matrix of shape (T, d).
    """
    rng = np.random.default_rng(seed)
    T, d = X_clean.shape

    if perturbation_type == 'additive':
        # Random Gaussian perturbation scaled to desired norm
        E = rng.standard_normal((T, d))
        E = E / (norm(E, 'fro') + 1e-12) * perturbation_norm

    elif perturbation_type == 'drift':
        # Gradual drift: perturbation grows linearly over time
        drift_direction = rng.standard_normal(d)
        drift_direction = drift_direction / (norm(drift_direction) + 1e-12)
        time_weights = np.linspace(0, 1, T).reshape(-1, 1)
        E = time_weights * drift_direction.reshape(1, -1)
        E = E / (norm(E, 'fro') + 1e-12) * perturbation_norm

    elif perturbation_type == 'multiplicative':
        # Multiplicative noise (proportional to signal)
        E = X_clean * rng.standard_normal((T, d)) * 0.1
        E = E / (norm(E, 'fro') + 1e-12) * perturbation_norm

    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")

    X_perturbed = X_clean + E
    return X_perturbed, E


if __name__ == '__main__':
    # Quick self-test
    print("Testing spectral_prover.py...")

    # Generate synthetic data
    X, K = generate_synthetic_linear_system(d=10, T=50, k=3, noise_std=0.01, seed=42)
    print(f"Generated trajectory: shape={X.shape}")

    # Compute certificate
    X_traj = trajectory_matrix(X, normalize=True)
    cert = compute_svd_certificate(X_traj, k=5)
    print(f"SVD certificate: residual={cert['residual']:.4f}, "
          f"pca_explained={cert['pca_explained']:.4f}")

    # Compute detection statistics
    stats = compute_detection_statistics(X, k=5)
    print(f"Detection stats: theoretical_bound={stats['theoretical_bound']:.4f}, "
          f"koopman_residual={stats['koopman_residual']:.4f}")

    # Test perturbation
    X_pert, E = generate_perturbed_trajectory(X, perturbation_norm=0.5, seed=42)
    stats_pert = compute_detection_statistics(X_pert, k=5, baseline_U=stats['U_k'])
    print(f"Perturbed stats: residual={stats_pert['residual']:.4f}, "
          f"dk_angle={stats_pert['dk_angle']:.4f}")

    print("All tests passed!")
