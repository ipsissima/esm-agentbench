"""Spectral certificate computation using PCA and Koopman proxy.

This module augments the certificate with conservative theoretical bounds
on reconstruction error using PCA tail estimates.
"""
from __future__ import annotations

import json
from typing import Dict, Iterable, List, Tuple

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


def segment_trace_by_jump(residuals: List[float], jump_threshold: float) -> List[Tuple[int, int]]:
    """Segment residual series whenever a jump exceeds ``jump_threshold``."""

    clean = [r if r is not None else 0.0 for r in residuals]
    if not clean:
        return [(0, 0)]
    segments: List[Tuple[int, int]] = []
    start = 0
    for idx in range(1, len(clean)):
        if abs(clean[idx] - clean[idx - 1]) > jump_threshold:
            segments.append((start, idx))
            start = idx
    segments.append((start, len(clean)))
    return segments


def _compute_certificate_core(X: np.ndarray, r: int) -> Dict[str, float]:
    T = X.shape[0]
    if T < 2 or X.size == 0:
        return _initial_empty_certificate()

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

    X0 = Z[:-1].T
    X1 = Z[1:].T

    gram = X0 @ X0.T
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
        "Z": Z,
    }


def compute_certificate(embeddings: Iterable[Iterable[float]], r: int = 10) -> Dict[str, float]:
    """Compute PCA reduction then finite-rank Koopman spectral summary."""

    X = _safe_array(embeddings)
    base = _compute_certificate_core(X, r)
    certificate = {k: v for k, v in base.items() if k != "Z"}

    Z = base.get("Z")
    if Z is None or not isinstance(Z, np.ndarray):
        return certificate

    # Optional segmented modeling to support pivot detection downstream.
    try:
        seg_residuals = residuals_from_Z(Z)
        segments = segment_trace_by_jump([float(v) for v in seg_residuals], jump_threshold=0.3)
        if len(segments) > 1:
            segment_meta = []
            for start, end in segments:
                if end - start < 2:
                    continue
                cert_seg = _compute_certificate_core(Z[start:end], r=min(r, end - start))
                segment_meta.append(
                    {
                        "start": start,
                        "end": end,
                        "pca_explained": cert_seg.get("pca_explained", 0.0),
                        "residual": cert_seg.get("residual", 0.0),
                    }
                )
            if segment_meta:
                certificate["segments"] = segment_meta
    except Exception:
        # Segmentation is optional; never fail certificate computation.
        pass

    return certificate


def residuals_from_Z(Z: np.ndarray) -> List[float]:
    """Compute residual magnitudes directly from reduced states."""

    if Z.shape[0] < 2:
        return [0.0] * Z.shape[0]
    X0 = Z[:-1].T
    X1 = Z[1:].T
    gram = X0 @ X0.T
    eps = 1e-12
    gram = gram + eps * np.eye(gram.shape[0])
    A = (X1 @ X0.T) @ pinv(gram)
    errs = X1 - A @ X0
    return [float(norm(errs[:, i]) / (norm(X1[:, i]) + eps)) for i in range(errs.shape[1])]


if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(0)
    X_demo = rng.standard_normal((20, 5))
    cert = compute_certificate(X_demo)
    print(json.dumps(cert, indent=2))
