#!/usr/bin/env python3
"""CLI for computing spectral certificates with verified kernel support.

This CLI provides opt-in access to:
- Per-step diagnostics (off-manifold ratio, residual norms)
- Subspace angle computation (sinTheta, E_over_gamma)
- Verified kernel integration (prototype or Docker)
- Certificate bundle creation

Usage:
    python -m certificates.make_certificate_cli trace.json [options]

Examples:
    # Basic certificate computation
    python -m certificates.make_certificate_cli trace.json

    # Export kernel input for verified computation
    python -m certificates.make_certificate_cli trace.json --export-kernel-input kernel_input.json

    # Run with verified kernel
    python -m certificates.make_certificate_cli trace.json --verify-with-kernel

    # Full pipeline with bundle
    python -m certificates.make_certificate_cli trace.json \
        --verify-with-kernel \
        --export-kernel-input kernel_input.json \
        --bundle-dir ./bundle

Environment Variables:
    ESM_KERNEL_IMAGE: Docker image for production kernel
    ESM_KERNEL_LOCAL_PY: Path to local prototype kernel
    ESM_CAPTURE_INTERNAL: Set to "1" to capture residual stream (requires model)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .make_certificate import (
    compute_certificate,
    compute_per_step_off_manifold,
    compute_per_step_residuals,
    compute_sin_theta,
    compute_E_norm,
    export_kernel_input,
    load_kernel_input,
    _fit_temporal_operator_ridge,
    _select_effective_rank,
)
from .kernel_client import (
    run_kernel,
    run_kernel_and_verify,
    get_kernel_diagnostics,
    KernelClientError,
)
from .cert_bundle import create_bundle, verify_bundle, BundleError

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
)
logger = logging.getLogger(__name__)


def load_trace(trace_path: str) -> Dict[str, Any]:
    """Load trace from JSON file.

    Parameters
    ----------
    trace_path : str
        Path to trace JSON file.

    Returns
    -------
    Dict[str, Any]
        Trace data with embeddings.
    """
    with open(trace_path, 'r') as f:
        data = json.load(f)
    return data


def extract_embeddings(trace: Dict[str, Any], psi_mode: str = "embedding") -> np.ndarray:
    """Extract embeddings from trace based on PSI mode.

    Parameters
    ----------
    trace : Dict[str, Any]
        Trace data.
    psi_mode : str
        PSI mode: "embedding" or "residual_stream".

    Returns
    -------
    np.ndarray
        Embedding matrix of shape (T, D).
    """
    if psi_mode == "residual_stream":
        # Look for internal_state field (requires ESM_CAPTURE_INTERNAL=1)
        if "internal_state" in trace:
            return np.array(trace["internal_state"])
        elif "residual_stream" in trace:
            return np.array(trace["residual_stream"])
        else:
            logger.warning(
                "No residual_stream found in trace. "
                "Set ESM_CAPTURE_INTERNAL=1 during generation to capture internal states. "
                "Falling back to embeddings."
            )

    # Default: use embeddings
    if "embeddings" in trace:
        return np.array(trace["embeddings"])
    elif "steps" in trace:
        # Extract embeddings from steps
        embeddings = []
        for step in trace["steps"]:
            if "embedding" in step:
                embeddings.append(step["embedding"])
            elif "state" in step and "embedding" in step["state"]:
                embeddings.append(step["state"]["embedding"])
        if embeddings:
            return np.array(embeddings)

    raise ValueError(
        "No embeddings found in trace. "
        "Trace must have 'embeddings' field or 'steps' with 'embedding' per step."
    )


def compute_enhanced_certificate(
    embeddings: np.ndarray,
    rank: int = 10,
    task_embedding: Optional[np.ndarray] = None,
    include_per_step: bool = True,
    include_sin_theta: bool = True,
    kernel_strict: bool = False,
    embedder_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute certificate with enhanced diagnostics.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix of shape (T, D).
    rank : int
        Target rank for SVD truncation.
    task_embedding : Optional[np.ndarray]
        Task embedding for semantic divergence.
    include_per_step : bool
        Include per-step diagnostics (off_ratio_t, r_norm_t).
    include_sin_theta : bool
        Include subspace angle summary (sinTheta, E_over_gamma).
    kernel_strict : bool
        Require verified kernel.
    embedder_id : Optional[str]
        Embedding model identifier.

    Returns
    -------
    Dict[str, Any]
        Enhanced certificate with diagnostics.
    """
    # Compute base certificate
    cert = compute_certificate(
        embeddings,
        r=rank,
        task_embedding=task_embedding,
        kernel_strict=kernel_strict,
        embedder_id=embedder_id,
    )

    T, D = embeddings.shape

    # Augment with bias
    X_aug = np.concatenate([embeddings, np.ones((T, 1))], axis=1)

    # Compute SVD for diagnostics
    U, S, Vt = np.linalg.svd(X_aug, full_matrices=False)
    r_eff, _ = _select_effective_rank(S, T, rank, X_aug.shape[1])
    r_eff = int(r_eff)  # Ensure r_eff is a plain Python int for slicing
    V_r = Vt[:r_eff, :].T  # (D+1, r_eff)

    # Project to reduced space
    Z = X_aug @ V_r

    # Fit temporal operator
    if Z.shape[0] >= 2:
        X0 = Z[:-1].T
        X1 = Z[1:].T
        A = _fit_temporal_operator_ridge(X0, X1, regularization=1e-6)
    else:
        A = None

    # Per-step diagnostics
    if include_per_step:
        off_ratios = compute_per_step_off_manifold(X_aug, V_r)
        if A is not None:
            r_norms = compute_per_step_residuals(Z, A)
        else:
            r_norms = [0.0] * T

        cert["per_step_diagnostics"] = {
            "off_ratio_t": off_ratios,
            "r_norm_t": r_norms,
            "description": "Per-step off-manifold ratio and residual norms",
        }

        # Summary statistics
        cert["off_ratio_max"] = float(max(off_ratios)) if off_ratios else 0.0
        cert["off_ratio_mean"] = float(np.mean(off_ratios)) if off_ratios else 0.0
        cert["r_norm_max"] = float(max(r_norms[:-1])) if len(r_norms) > 1 else 0.0
        cert["r_norm_mean"] = float(np.mean(r_norms[:-1])) if len(r_norms) > 1 else 0.0

    # Subspace angle summary
    if include_sin_theta and A is not None:
        # Compute subspace angles between trajectory subspace and identity
        U_traj = Vt[:r_eff, :].T  # Trajectory subspace
        U_identity = np.eye(U_traj.shape[0], min(r_eff, U_traj.shape[0]))

        sin_max, sin_fro = compute_sin_theta(U_traj, U_identity)

        # Compute E_over_gamma (Wedin condition)
        # gamma = singular gap
        gamma = float(S[r_eff - 1] - S[r_eff]) if len(S) > r_eff else float(S[-1])
        gamma = max(gamma, 1e-10)  # Avoid division by zero

        # E_norm = operator perturbation (residual-based estimate)
        if A is not None:
            E_norm_estimate = float(cert.get("residual", 0.0))
        else:
            E_norm_estimate = 0.0

        E_over_gamma = E_norm_estimate / gamma

        cert["sinTheta"] = {
            "sin_max": sin_max,
            "sin_frobenius": sin_fro,
            "description": "Subspace angles between trajectory and reference subspace",
        }
        cert["E_over_gamma"] = E_over_gamma
        cert["gamma"] = gamma
        cert["wedin_condition"] = {
            "E_over_gamma": E_over_gamma,
            "gamma": gamma,
            "sin_theta_max": sin_max,
            "bound_holds": E_over_gamma < 1.0,  # Wedin requires E/gamma < 1
            "description": "Wedin's theorem: sin(theta) <= E/gamma when E/gamma < 1",
        }

    return cert


def run_with_kernel_verification(
    trace_path: str,
    embeddings: np.ndarray,
    kernel_input_path: Optional[str] = None,
    kernel_output_path: Optional[str] = None,
    kernel_mode: str = "prototype",
    precision_bits: int = 128,
) -> Dict[str, Any]:
    """Run certificate computation with verified kernel.

    Parameters
    ----------
    trace_path : str
        Path to original trace.
    embeddings : np.ndarray
        Embedding matrix.
    kernel_input_path : Optional[str]
        Path to export kernel input.
    kernel_output_path : Optional[str]
        Path to export kernel output.
    kernel_mode : str
        Kernel mode: "prototype", "arb", "mpfi".
    precision_bits : int
        Precision bits for interval arithmetic.

    Returns
    -------
    Dict[str, Any]
        Certificate with kernel verification results.
    """
    import hashlib
    import tempfile

    T, D = embeddings.shape

    # Augment with bias
    X_aug = np.concatenate([embeddings, np.ones((T, 1))], axis=1)

    # Generate trace ID
    trace_id = hashlib.sha256(X_aug.tobytes()).hexdigest()[:16]

    # Export kernel input
    if kernel_input_path is None:
        tmpdir = tempfile.mkdtemp(prefix="esm_kernel_")
        kernel_input_path = os.path.join(tmpdir, "kernel_input.json")

    export_kernel_input(
        X_aug=X_aug,
        trace_id=trace_id,
        output_path=kernel_input_path,
        precision_bits=precision_bits,
        kernel_mode=kernel_mode,
    )
    logger.info(f"Exported kernel input to {kernel_input_path}")

    # Run kernel
    try:
        kernel_output = run_kernel_and_verify(
            kernel_input_path,
            output_path=kernel_output_path,
            precision_bits=precision_bits,
            mode=kernel_mode,
        )
        kernel_verified = True
        kernel_error = None
        logger.info("Kernel verification PASSED")
    except KernelClientError as e:
        kernel_output = {}
        kernel_verified = False
        kernel_error = str(e)
        logger.warning(f"Kernel verification FAILED: {e}")

    return {
        "kernel_verified": kernel_verified,
        "kernel_error": kernel_error,
        "kernel_output": kernel_output,
        "kernel_input_path": kernel_input_path,
        "trace_id": trace_id,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute spectral certificates with verified kernel support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "trace",
        help="Path to trace JSON file with embeddings",
    )

    parser.add_argument(
        "--rank", "-r",
        type=int,
        default=10,
        help="Maximum rank for SVD truncation (default: 10)",
    )

    parser.add_argument(
        "--output", "-o",
        help="Path to output certificate JSON (default: stdout)",
    )

    # Kernel input/output export
    parser.add_argument(
        "--export-kernel-input",
        metavar="PATH",
        help="Export kernel input JSON to specified path",
    )

    parser.add_argument(
        "--export-kernel-output",
        metavar="PATH",
        help="Path for kernel output JSON (with --verify-with-kernel)",
    )

    # PSI mode (embedding vs residual_stream)
    parser.add_argument(
        "--psi",
        choices=["embedding", "residual_stream"],
        default="embedding",
        help="Projection state indicator: 'embedding' (default) or 'residual_stream' (requires ESM_CAPTURE_INTERNAL=1)",
    )

    # Verified kernel
    parser.add_argument(
        "--verify-with-kernel",
        action="store_true",
        help="Run verified kernel for interval-bounded computation",
    )

    parser.add_argument(
        "--kernel-mode",
        choices=["prototype", "arb", "mpfi"],
        default="prototype",
        help="Kernel mode: 'prototype' (mpmath), 'arb' (Docker ARB), 'mpfi' (Docker MPFI)",
    )

    parser.add_argument(
        "--precision",
        type=int,
        default=128,
        help="Precision bits for interval arithmetic (default: 128)",
    )

    # Bundle creation
    parser.add_argument(
        "--bundle-dir",
        metavar="DIR",
        help="Create signed certificate bundle in specified directory",
    )

    parser.add_argument(
        "--gpg-key",
        metavar="KEY",
        help="GPG key ID for signing bundle (optional)",
    )

    # Diagnostics options
    parser.add_argument(
        "--no-per-step",
        action="store_true",
        help="Disable per-step diagnostics (off_ratio_t, r_norm_t)",
    )

    parser.add_argument(
        "--no-sin-theta",
        action="store_true",
        help="Disable subspace angle summary (sinTheta, E_over_gamma)",
    )

    # Misc
    parser.add_argument(
        "--embedder-id",
        help="Embedding model identifier for audit trail",
    )

    parser.add_argument(
        "--kernel-strict",
        action="store_true",
        help="Fail if verified kernel is unavailable",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Show kernel diagnostics and exit",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Show diagnostics and exit
    if args.diagnostics:
        diag = get_kernel_diagnostics()
        print(json.dumps(diag, indent=2))
        return 0

    # Load trace
    try:
        trace = load_trace(args.trace)
    except FileNotFoundError:
        logger.error(f"Trace file not found: {args.trace}")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in trace file: {e}")
        return 1

    # Extract embeddings
    try:
        embeddings = extract_embeddings(trace, psi_mode=args.psi)
    except ValueError as e:
        logger.error(str(e))
        return 1

    logger.info(f"Loaded embeddings: shape {embeddings.shape}")

    # Extract task embedding if available
    task_embedding = None
    if "task_embedding" in trace:
        task_embedding = np.array(trace["task_embedding"])
    elif "task" in trace and "embedding" in trace["task"]:
        task_embedding = np.array(trace["task"]["embedding"])

    # Compute enhanced certificate
    cert = compute_enhanced_certificate(
        embeddings,
        rank=args.rank,
        task_embedding=task_embedding,
        include_per_step=not args.no_per_step,
        include_sin_theta=not args.no_sin_theta,
        kernel_strict=args.kernel_strict,
        embedder_id=args.embedder_id,
    )

    # Run with verified kernel if requested
    kernel_result = None
    if args.verify_with_kernel:
        kernel_result = run_with_kernel_verification(
            trace_path=args.trace,
            embeddings=embeddings,
            kernel_input_path=args.export_kernel_input,
            kernel_output_path=args.export_kernel_output,
            kernel_mode=args.kernel_mode,
            precision_bits=args.precision,
        )
        cert["kernel_verification"] = kernel_result

    # Export kernel input (without running kernel)
    elif args.export_kernel_input:
        import hashlib
        T, D = embeddings.shape
        X_aug = np.concatenate([embeddings, np.ones((T, 1))], axis=1)
        trace_id = hashlib.sha256(X_aug.tobytes()).hexdigest()[:16]

        export_kernel_input(
            X_aug=X_aug,
            trace_id=trace_id,
            output_path=args.export_kernel_input,
            precision_bits=args.precision,
            kernel_mode=args.kernel_mode,
            embedder_id=args.embedder_id,
        )
        logger.info(f"Exported kernel input to {args.export_kernel_input}")
        cert["kernel_input_exported"] = args.export_kernel_input

    # Create bundle if requested
    if args.bundle_dir:
        import tempfile

        # Write certificate to temp file
        cert_path = os.path.join(args.bundle_dir, "certificate.json")
        os.makedirs(args.bundle_dir, exist_ok=True)

        with open(cert_path, 'w') as f:
            json.dump(cert, f, indent=2, default=str)

        try:
            bundle_path = create_bundle(
                bundle_dir=args.bundle_dir,
                trace_path=args.trace,
                kernel_input_path=args.export_kernel_input,
                kernel_output_path=args.export_kernel_output,
                certificate_path=cert_path,
                embedder_id=args.embedder_id,
                kernel_mode=args.kernel_mode if args.verify_with_kernel else None,
                gpg_key=args.gpg_key,
            )
            logger.info(f"Created bundle at {bundle_path}")
            cert["bundle_path"] = bundle_path
        except BundleError as e:
            logger.warning(f"Failed to create bundle: {e}")

    # Output certificate
    cert_json = json.dumps(cert, indent=2, default=str)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(cert_json)
        logger.info(f"Certificate written to {args.output}")
    else:
        print(cert_json)

    # Return exit code based on theoretical bound
    bound = cert.get("theoretical_bound", float("inf"))
    if bound > 0.5:
        logger.warning(f"Certificate bound {bound:.4f} exceeds threshold 0.5")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
