"""Core certificate computation with optional kernel verification.

This module provides the main entry point for computing spectral certificates
from trace data. It handles:
- Embedding extraction from various trace formats
- Certificate computation via certificates.make_certificate
- Optional kernel verification via ports.kernel.KernelClientPort

Shape conventions:
- Embedding matrices: (T, D) where T = time steps, D = embedding dimension
- Time is the first axis (rows), features are the second axis (columns)
- All computations use float64 (np.float64) for numerical stability
- Augmented embeddings X_aug: (T, D+1) with affine bias term

The compute_certificate_from_trace function is the canonical interface for
certificate generation from agent execution traces.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from typing import Any, Dict, Mapping, Optional

import numpy as np

from certificates.make_certificate import compute_certificate, export_kernel_input
from ports.kernel import KernelClientPort


class TraceEmbeddingError(ValueError):
    """Raised when embeddings cannot be extracted from a trace."""


def extract_embeddings(trace: Mapping[str, Any], psi_mode: str = "embedding") -> np.ndarray:
    """Extract embeddings from a trace payload.

    Parameters
    ----------
    trace : Mapping[str, Any]
        Trace data containing embeddings or steps with embeddings.
    psi_mode : str
        Mode for extraction:
        - "embedding": Uses 'embeddings' field or extracts from 'steps'
        - "residual_stream": Uses 'internal_state' or 'residual_stream' if available

    Returns
    -------
    np.ndarray
        Embedding matrix of shape (T, D) where T is number of time steps
        and D is embedding dimension. Dtype is float64.

    Raises
    ------
    TraceEmbeddingError
        If no embeddings can be extracted from the trace.

    Notes
    -----
    Shape convention: Time is the first axis (rows), features second (columns).
    This follows the (T, D) convention used throughout the certificate pipeline.
    """
    if psi_mode == "residual_stream":
        if "internal_state" in trace:
            return np.array(trace["internal_state"], dtype=float)
        if "residual_stream" in trace:
            return np.array(trace["residual_stream"], dtype=float)

    if "embeddings" in trace:
        return np.array(trace["embeddings"], dtype=float)

    if "steps" in trace:
        embeddings = []
        for step in trace["steps"]:
            if "embedding" in step:
                embeddings.append(step["embedding"])
            elif "state" in step and "embedding" in step["state"]:
                embeddings.append(step["state"]["embedding"])
        if embeddings:
            return np.array(embeddings, dtype=float)

    raise TraceEmbeddingError(
        "No embeddings found in trace. Expected 'embeddings' or 'steps' with embeddings."
    )


def _hash_matrix_bytes(matrix: np.ndarray) -> str:
    """Hash a matrix using SHA256."""
    data = matrix.astype(np.float64).tobytes()
    return hashlib.sha256(data).hexdigest()


def compute_certificate_from_trace(
    trace: Mapping[str, Any],
    *,
    psi_mode: str = "embedding",
    rank: int = 10,
    task_embedding: Optional[np.ndarray] = None,
    embedder_id: Optional[str] = None,
    kernel: Optional[KernelClientPort] = None,
    kernel_mode: str = "prototype",
    precision_bits: int = 128,
) -> Dict[str, Any]:
    """Compute a certificate from a trace with optional kernel verification.

    This is the canonical interface for generating spectral certificates from
    agent execution traces. It extracts embeddings, computes the certificate
    via SVD-based analysis, and optionally verifies results with a kernel.

    Parameters
    ----------
    trace : Mapping[str, Any]
        Trace payload containing embeddings or steps with embeddings.
    psi_mode : str, optional
        Embedding extraction mode (default: "embedding").
    rank : int, optional
        Rank for SVD truncation (default: 10).
    task_embedding : Optional[np.ndarray], optional
        Task embedding for semantic divergence computation. Should be shape (D,)
        matching the embedding dimension.
    embedder_id : Optional[str], optional
        Embedder identifier for audit trails.
    kernel : Optional[KernelClientPort], optional
        Kernel adapter for verification. If provided, runs kernel verification
        and includes kernel output in the certificate.
    kernel_mode : str, optional
        Kernel execution mode: "prototype", "arb", or "mpfi" (default: "prototype").
    precision_bits : int, optional
        Interval arithmetic precision for kernel verification (default: 128).

    Returns
    -------
    Dict[str, Any]
        Certificate payload containing:
        - 'theoretical_bound': float, the certified theoretical bound
        - 'residual': float, the prediction residual
        - 'tail_energy': float, energy not captured by rank-r truncation
        - 'semantic_divergence': float, deviation from task embedding
        - 'lipschitz_margin': float, embedding Lipschitz stability
        - 'kernel_output': Dict (if kernel provided), verification results
        - 'kernel_trace_id': str (if kernel provided), trace identifier

    Notes
    -----
    Shape conventions:
    - Input embeddings: (T, D) with T time steps, D dimensions
    - Task embedding: (D,) vector
    - All arrays converted to float64 for numerical stability
    """
    embeddings = extract_embeddings(trace, psi_mode=psi_mode)
    certificate = compute_certificate(
        embeddings,
        r=rank,
        task_embedding=task_embedding,
        embedder_id=embedder_id,
        kernel_strict=False,
    )

    if kernel is None:
        return certificate

    T, D = embeddings.shape
    X_aug = np.concatenate([embeddings, np.ones((T, 1))], axis=1)
    trace_id = _hash_matrix_bytes(X_aug)

    with tempfile.TemporaryDirectory(prefix="esm_kernel_") as tmpdir:
        kernel_input_path = f"{tmpdir}/kernel_input.json"
        export_kernel_input(
            X_aug,
            trace_id,
            kernel_input_path,
            embedder_id=embedder_id,
            rank=rank,
            precision_bits=precision_bits,
            kernel_mode=kernel_mode,
        )

        kernel_output = kernel.run_kernel_and_verify(
            kernel_input_path,
            precision_bits=precision_bits,
            mode=kernel_mode,
        )

    certificate = dict(certificate)
    certificate["kernel_output"] = kernel_output
    certificate["kernel_trace_id"] = trace_id
    return certificate
