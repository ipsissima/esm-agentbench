"""Core certificate computation with optional kernel verification."""
from __future__ import annotations

import hashlib
import json
import tempfile
from typing import Any, Dict, Mapping, Optional

import numpy as np

from certificates.make_certificate import compute_certificate, export_kernel_input
from ports.kernel import KernelPort


class TraceEmbeddingError(ValueError):
    """Raised when embeddings cannot be extracted from a trace."""


def extract_embeddings(trace: Mapping[str, Any], psi_mode: str = "embedding") -> np.ndarray:
    """Extract embeddings from a trace payload.

    Parameters
    ----------
    trace : Mapping[str, Any]
        Trace data.
    psi_mode : str
        Mode for extraction. "embedding" uses embeddings, "residual_stream" uses
        internal state/residual stream if available.

    Returns
    -------
    np.ndarray
        Embedding matrix of shape (T, D).
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
    kernel: Optional[KernelPort] = None,
    kernel_mode: str = "prototype",
    precision_bits: int = 128,
) -> Dict[str, Any]:
    """Compute a certificate from a trace with optional kernel verification.

    Parameters
    ----------
    trace : Mapping[str, Any]
        Trace payload containing embeddings or steps.
    psi_mode : str
        Embedding selection mode.
    rank : int
        Rank for SVD truncation.
    task_embedding : Optional[np.ndarray]
        Task embedding used for semantic divergence.
    embedder_id : Optional[str]
        Embedder identifier for audit trails.
    kernel : Optional[KernelPort]
        Kernel adapter to verify results.
    kernel_mode : str
        Kernel mode (prototype/arb/mpfi).
    precision_bits : int
        Interval arithmetic precision.

    Returns
    -------
    Dict[str, Any]
        Certificate payload with optional kernel output.
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
