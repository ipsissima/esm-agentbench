"""Spectral certificate computation using PCA and Koopman proxy.

This module augments the certificate with conservative theoretical bounds
on reconstruction error using PCA tail estimates.
"""
from __future__ import annotations

import json
from typing import Dict, Iterable

import numpy as np
from numpy.linalg import norm, pinv
from sklearn.decomposition import PCA


def _safe_array(embeddings: Iterable[Iterable[float]]) -> np.ndarray:
    """Convert embeddings to a 2D float array with defensive defaults."""
    X = np.array(list(embeddings), dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


def _initial_empty_certificate() -> Dict[str, float]:
    """Return a conservative empty certificate when data is insufficient."""
    return {
        "pca_explained": 0.0,
        "max_eig": float("nan"),
        "spectral_gap": 0.0,
        "residual": 1.0,
        "pca_tail_estimate": 1.0,
        "theoretical_bound": 2.0,
    }


def compute_certificate(embeddings: Iterable[Iterable[float]], r: int = 10) -> Dict[str, float]:
    """Compute PCA reduction then finite-rank Koopman spectral summary.

    The routine follows three stages: (1) augment embeddings with a bias term
    to capture affine drift; (2) fit PCA to reduce dimensionality; (3) compute
    a finite-rank Koopman proxy via least-squares and measure reconstruction
    residual. The certificate additionally reports a conservative theoretical
    bound defined as ``residual + pca_tail_estimate`` where ``pca_tail_estimate``
    is the fraction of variance omitted by PCA.

    Parameters
    ----------
    embeddings: iterable of iterables
        Sequence of embedding vectors. Shape (T, d).
    r: int
        Maximum reduced dimension for PCA.

    Returns
    -------
    dict
        Dictionary with pca_explained, max_eig, spectral_gap, residual,
        pca_tail_estimate, theoretical_bound.
    """

    X = _safe_array(embeddings)
    T = X.shape[0]
    if T < 2 or X.size == 0:
        return _initial_empty_certificate()

    # Augment with a bias term so affine drifts remain representable linearly.
    X_aug = np.concatenate([X, np.ones((T, 1))], axis=1)

    r_eff = max(1, min(r, T - 1, X_aug.shape[1]))
    pca = PCA(n_components=r_eff, svd_solver="auto", random_state=0)
    pca.fit(X_aug)
    Z = X_aug @ pca.components_.T
    Z = np.asarray(Z, dtype=float)

    pca_explained = float(np.clip(np.sum(pca.explained_variance_ratio_), 0.0, 1.0))
    pca_tail_estimate = float(max(0.0, 1.0 - pca_explained))

    if Z.shape[0] < 2:
        certificate = _initial_empty_certificate()
        certificate.update(
            {
                "pca_explained": pca_explained,
                "pca_tail_estimate": pca_tail_estimate,
            }
        )
        certificate["theoretical_bound"] = float(certificate["residual"] + pca_tail_estimate)
        return certificate

    X0 = Z[:-1].T  # shape (r_eff, T-1)
    X1 = Z[1:].T

    gram = X0 @ X0.T
    # Small Tikhonov regularization for numerical stability.
    eps = 1e-12
    gram = gram + eps * np.eye(gram.shape[0])
    A = (X1 @ X0.T) @ pinv(gram)

    eigs = np.linalg.eigvals(A)
    mags = np.abs(eigs)
    mags_sorted = np.sort(mags)[::-1]
    max_eig = float(mags_sorted[0]) if mags_sorted.size else float("nan")
    spectral_gap = float(mags_sorted[0] - mags_sorted[1]) if mags_sorted.size > 1 else 0.0

    residual = float(norm(X1 - A @ X0, ord="fro") / (norm(X1, ord="fro") + eps))
    theoretical_bound = float(residual + pca_tail_estimate)

    return {
        "pca_explained": pca_explained,
        "max_eig": max_eig,
        "spectral_gap": spectral_gap,
        "residual": residual,
        "pca_tail_estimate": pca_tail_estimate,
        "theoretical_bound": theoretical_bound,
    }


if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(0)
    X_demo = rng.standard_normal((20, 5))
    cert = compute_certificate(X_demo)
    print(json.dumps(cert, indent=2))
