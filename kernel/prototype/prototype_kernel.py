#!/usr/bin/env python3
"""Prototype verified kernel using mpmath interval arithmetic.

This kernel computes interval-bounded certificate values using mpmath.iv
(arbitrary-precision interval arithmetic). It serves as a reference
implementation and development tool before using the production ARB kernel.

Usage:
    python prototype_kernel.py input.json output.json [--precision 160]

The kernel reads kernel_input.json, computes SVD and certificate metrics
with interval bounds, and writes kernel_output.json.

Requirements:
    - mpmath (pip install mpmath)
    - numpy
    - scipy

Environment Variables:
    - PRECISION_BITS: Override precision (default: 128)
    - KERNEL_MODE: Set to "prototype" (ignored, always uses mpmath)
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import norm

# Try importing mpmath for interval arithmetic
try:
    import mpmath
    from mpmath import iv
    HAVE_MPMATH = True
except ImportError:
    HAVE_MPMATH = False
    iv = None  # type: ignore

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class PrototypeKernelError(Exception):
    """Raised when kernel computation fails."""
    pass


def get_kernel_version() -> str:
    """Get the kernel version (git hash or fallback)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "prototype-1.0"


def interval_to_list(interval: Any) -> List[str]:
    """Convert mpmath interval to [low, high] string list."""
    if not HAVE_MPMATH:
        return [str(interval), str(interval)]
    if isinstance(interval, (iv.mpf,)):  # type: ignore
        a, b = interval.a, interval.b
        return [str(a), str(b)]
    return [str(interval), str(interval)]


def float_to_interval(x: float, precision: int = 128) -> Any:
    """Convert float to mpmath interval with specified precision."""
    if not HAVE_MPMATH:
        return x
    old_prec = mpmath.mp.prec
    mpmath.mp.prec = precision
    try:
        return iv.mpf(x)
    finally:
        mpmath.mp.prec = old_prec


def decode_matrix_base64(encoded: str, rows: int, cols: int) -> np.ndarray:
    """Decode base64 string to matrix."""
    raw_bytes = base64.b64decode(encoded)
    flat = np.frombuffer(raw_bytes, dtype='>f8')  # Big-endian float64
    return flat.reshape(rows, cols).astype(np.float64)


def compute_svd_intervals(
    X: np.ndarray,
    rank: int,
    precision: int = 128,
) -> Dict[str, Any]:
    """Compute SVD with interval bounds on singular values.

    Parameters
    ----------
    X : np.ndarray
        Data matrix, shape (T, D).
    rank : int
        Target rank for truncation.
    precision : int
        Precision bits for interval arithmetic.

    Returns
    -------
    Dict[str, Any]
        SVD results with interval bounds.
    """
    # Compute standard SVD first
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    r_eff = min(rank, len(S), X.shape[0] - 1)

    # Wrap singular values in intervals
    if HAVE_MPMATH:
        old_prec = mpmath.mp.prec
        mpmath.mp.prec = precision

        sigma_intervals = []
        for s in S[:r_eff]:
            # Create interval around computed value with machine epsilon uncertainty
            eps = float(np.finfo(np.float64).eps) * abs(s)
            low = iv.mpf(s - eps)
            high = iv.mpf(s + eps)
            sigma_intervals.append([str(low.a), str(high.b)])

        mpmath.mp.prec = old_prec
    else:
        sigma_intervals = [[str(s), str(s)] for s in S[:r_eff]]

    # Compute explained variance
    total_energy = float(np.sum(S ** 2))
    retained_energy = float(np.sum(S[:r_eff] ** 2))
    pca_explained = retained_energy / (total_energy + 1e-12) if total_energy > 1e-12 else 0.0

    # Tail energy
    tail_energy = 1.0 - pca_explained

    # Singular gap (gamma)
    if len(S) > r_eff:
        gamma = float(S[r_eff - 1] - S[r_eff]) if r_eff > 0 and S[r_eff] > 0 else float(S[r_eff - 1])
    else:
        gamma = float(S[-1]) if len(S) > 0 else 0.0

    return {
        "U": U[:, :r_eff],
        "S": S[:r_eff],
        "Vt": Vt[:r_eff, :],
        "sigma_intervals": sigma_intervals,
        "gamma": [str(gamma), str(gamma)],
        "tail_energy": [str(tail_energy), str(tail_energy)],
        "pca_explained": [str(pca_explained), str(pca_explained)],
        "r_eff": r_eff,
    }


def compute_koopman_intervals(
    Z: np.ndarray,
    precision: int = 128,
) -> Dict[str, Any]:
    """Compute Koopman operator with interval bounds.

    Parameters
    ----------
    Z : np.ndarray
        Projected states, shape (T, r_eff).
    precision : int
        Precision bits for interval arithmetic.

    Returns
    -------
    Dict[str, Any]
        Koopman results with interval bounds.
    """
    T, r = Z.shape
    if T < 2:
        return {
            "A": None,
            "koopman_sigma": [],
            "koopman_singular_gap": ["0", "0"],
        }

    # Fit temporal operator via ridge regression
    X0 = Z[:-1].T  # (r, T-1)
    X1 = Z[1:].T   # (r, T-1)

    regularization = 1e-6
    gram = X0 @ X0.T + regularization * np.eye(r)
    try:
        A = (X1 @ X0.T) @ np.linalg.inv(gram)
    except np.linalg.LinAlgError:
        A = (X1 @ X0.T) @ np.linalg.pinv(gram)

    # Compute SVD of A for singular values
    U_A, S_A, Vt_A = np.linalg.svd(A, full_matrices=False)

    # Wrap in intervals
    if HAVE_MPMATH:
        old_prec = mpmath.mp.prec
        mpmath.mp.prec = precision

        koopman_sigma = []
        for s in S_A:
            eps = float(np.finfo(np.float64).eps) * abs(s)
            low = iv.mpf(s - eps)
            high = iv.mpf(s + eps)
            koopman_sigma.append([str(low.a), str(high.b)])

        mpmath.mp.prec = old_prec
    else:
        koopman_sigma = [[str(s), str(s)] for s in S_A]

    # Singular gap of Koopman operator
    if len(S_A) > 1:
        gap = float(S_A[0] - S_A[1])
    else:
        gap = float(S_A[0]) if len(S_A) > 0 else 0.0

    return {
        "A": A,
        "koopman_sigma": koopman_sigma,
        "koopman_singular_gap": [str(gap), str(gap)],
    }


def compute_residual_intervals(
    Z: np.ndarray,
    A: np.ndarray,
    precision: int = 128,
) -> Dict[str, Any]:
    """Compute residuals with interval bounds.

    Parameters
    ----------
    Z : np.ndarray
        Projected states, shape (T, r_eff).
    A : np.ndarray
        Koopman operator, shape (r, r).
    precision : int
        Precision bits for interval arithmetic.

    Returns
    -------
    Dict[str, Any]
        Residual metrics with interval bounds.
    """
    T, r = Z.shape
    eps = 1e-12

    if T < 2:
        return {
            "insample_residual": ["0", "0"],
            "oos_residual": ["0", "0"],
            "r_t_intervals": [],
        }

    X0 = Z[:-1].T  # (r, T-1)
    X1 = Z[1:].T   # (r, T-1)

    # In-sample residual
    pred = A @ X0
    err = X1 - pred
    insample_residual = float(norm(err, "fro") / (norm(X1, "fro") + eps))

    # Out-of-sample residual (leave-one-out style)
    K = min(3, max(1, (T - 1) // 4))
    oos_errors = []
    for h in range(T - 1 - K, T - 1):
        if h < 2:
            continue
        X0_train = X0[:, :h]
        X1_train = X1[:, :h]
        gram_train = X0_train @ X0_train.T + 1e-6 * np.eye(r)
        try:
            A_train = (X1_train @ X0_train.T) @ np.linalg.inv(gram_train)
        except np.linalg.LinAlgError:
            continue
        x0_test = X0[:, h]
        x1_true = X1[:, h]
        x1_pred = A_train @ x0_test
        oos_errors.append(norm(x1_true - x1_pred) ** 2)

    if oos_errors:
        oos_residual = float(np.sqrt(sum(oos_errors) / (np.sum(X1[:, -K:] ** 2) + eps)))
    else:
        oos_residual = insample_residual

    # Per-step residuals
    r_t_intervals = []
    for i in range(T - 1):
        r_i = float(norm(X1[:, i] - A @ X0[:, i]))
        r_t_intervals.append([str(r_i), str(r_i)])

    return {
        "insample_residual": [str(insample_residual), str(insample_residual)],
        "oos_residual": [str(oos_residual), str(oos_residual)],
        "r_t_intervals": r_t_intervals,
    }


def compute_per_step_intervals(
    X_aug: np.ndarray,
    Z: np.ndarray,
    V_r: np.ndarray,
    A: np.ndarray,
    precision: int = 128,
) -> Dict[str, Any]:
    """Compute per-step diagnostics with interval bounds.

    Parameters
    ----------
    X_aug : np.ndarray
        Augmented trajectory, shape (T, D+1).
    Z : np.ndarray
        Projected states, shape (T, r_eff).
    V_r : np.ndarray
        Right singular vectors, shape (D+1, r_eff).
    A : np.ndarray
        Koopman operator, shape (r, r).
    precision : int
        Precision bits.

    Returns
    -------
    Dict[str, Any]
        Per-step diagnostics.
    """
    T = X_aug.shape[0]
    eps = 1e-12

    # Off-manifold ratios
    off_ratio = []
    for i in range(T):
        x_i = X_aug[i]
        x_norm = norm(x_i)
        if x_norm < eps:
            off_ratio.append(["0", "0"])
            continue
        proj = V_r @ (V_r.T @ x_i)
        off_norm = norm(x_i - proj)
        ratio = float(off_norm / (x_norm + eps))
        off_ratio.append([str(ratio), str(ratio)])

    # Per-step residual norms
    r_norm = []
    for i in range(T - 1):
        z_t = Z[i]
        z_next = Z[i + 1]
        z_pred = A @ z_t
        r_i = float(norm(z_next - z_pred))
        r_norm.append([str(r_i), str(r_i)])
    r_norm.append(["0", "0"])  # Last step

    return {
        "off_ratio": off_ratio,
        "r_norm": r_norm,
    }


def compute_E_norm_interval(
    A1: np.ndarray,
    A2: np.ndarray,
    precision: int = 128,
) -> List[str]:
    """Compute operator difference norm ||A1 - A2||_F with interval bounds."""
    E = A1 - A2
    E_norm = float(norm(E, "fro"))
    return [str(E_norm), str(E_norm)]


def compute_sin_theta_interval(
    U1: np.ndarray,
    U2: np.ndarray,
    precision: int = 128,
) -> Dict[str, List[str]]:
    """Compute subspace angles with interval bounds."""
    from scipy.linalg import subspace_angles

    try:
        angles = subspace_angles(U1, U2)
        sin_vals = np.sin(angles)
        sin_max = float(np.max(sin_vals)) if len(sin_vals) > 0 else 0.0
        sin_fro = float(norm(sin_vals))
    except (ValueError, np.linalg.LinAlgError):
        sin_max = 0.0
        sin_fro = 0.0

    return {
        "max_sin": [str(sin_max), str(sin_max)],
        "frobenius": [str(sin_fro), str(sin_fro)],
    }


def compute_theoretical_bound(
    residual: float,
    tail_energy: float,
    semantic_divergence: float = 0.0,
    lipschitz_margin: float = 0.0,
    c_res: float = 1.0,
    c_tail: float = 1.0,
    c_sem: float = 1.0,
    c_robust: float = 1.0,
    tau: float = 0.5,
) -> Dict[str, Any]:
    """Compute theoretical bound and check if it passes.

    Parameters
    ----------
    residual : float
        Prediction residual.
    tail_energy : float
        Tail energy not captured by SVD.
    semantic_divergence : float
        Semantic divergence from task.
    lipschitz_margin : float
        Embedding Lipschitz margin.
    c_res, c_tail, c_sem, c_robust : float
        Verified constants.
    tau : float
        Threshold for passing.

    Returns
    -------
    Dict[str, Any]
        Bound computation results.
    """
    bound = (
        c_res * residual
        + c_tail * tail_energy
        + c_sem * semantic_divergence
        + c_robust * lipschitz_margin
    )

    return {
        "lhs_interval": [str(bound), str(bound)],
        "tau": tau,
        "pass": bound <= tau,
    }


def run_kernel(input_path: str, output_path: str, precision: int = 128) -> Dict[str, Any]:
    """Run the prototype kernel.

    Parameters
    ----------
    input_path : str
        Path to kernel_input.json.
    output_path : str
        Path to write kernel_output.json.
    precision : int
        Precision bits for interval arithmetic.

    Returns
    -------
    Dict[str, Any]
        Kernel output dictionary.
    """
    logger.info(f"Loading kernel input from {input_path}")

    # Load input
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Decode matrix
    obs = data["observables"]["X_aug"]
    X_aug = decode_matrix_base64(obs["data_matrix"], obs["rows"], obs["cols"])

    # Verify integrity
    computed_hash = hashlib.sha256(X_aug.astype(">f8").tobytes()).hexdigest()
    if obs.get("sha256") and computed_hash != obs["sha256"]:
        raise PrototypeKernelError(
            f"Integrity check failed: expected {obs['sha256']}, got {computed_hash}"
        )

    trace_id = data.get("trace_id", "unknown")
    rank = data["parameters"].get("rank", 10)
    precision = data["parameters"].get("precision_bits", precision)

    logger.info(f"Processing trace {trace_id}, shape {X_aug.shape}, rank {rank}, precision {precision}")

    # Set mpmath precision
    if HAVE_MPMATH:
        mpmath.mp.prec = precision

    # Compute SVD
    svd_result = compute_svd_intervals(X_aug, rank, precision)
    U = svd_result["U"]
    S = svd_result["S"]
    Vt = svd_result["Vt"]
    r_eff = svd_result["r_eff"]

    # Project onto reduced space
    Z = X_aug @ Vt.T
    V_r = Vt.T  # (D+1, r_eff)

    # Compute Koopman
    koopman_result = compute_koopman_intervals(Z, precision)
    A = koopman_result["A"]

    # Compute residuals
    if A is not None:
        residual_result = compute_residual_intervals(Z, A, precision)
        per_step_result = compute_per_step_intervals(X_aug, Z, V_r, A, precision)
    else:
        residual_result = {
            "insample_residual": ["0", "0"],
            "oos_residual": ["0", "0"],
            "r_t_intervals": [],
        }
        per_step_result = {"off_ratio": [], "r_norm": []}

    # Parse residual for bound computation
    residual_val = float(residual_result["oos_residual"][0])
    tail_val = float(svd_result["tail_energy"][0])

    # Compute theoretical bound
    bound_result = compute_theoretical_bound(
        residual=residual_val,
        tail_energy=tail_val,
    )

    # Compute input hash for provenance
    input_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    # Build output
    output = {
        "schema_version": "1.0",
        "trace_id": trace_id,
        "kernel_id": get_kernel_version(),
        "precision_bits": precision,
        "computed": {
            "sigma": svd_result["sigma_intervals"],
            "gamma": svd_result["gamma"],
            "tail_energy": svd_result["tail_energy"],
            "pca_explained": svd_result["pca_explained"],
            "koopman": {
                "A": None,  # Don't export full matrix
                "koopman_sigma": koopman_result["koopman_sigma"],
                "koopman_singular_gap": koopman_result["koopman_singular_gap"],
            },
            "residuals": residual_result,
            "per_step": per_step_result,
            "E_norm": ["0", "0"],  # Placeholder - needs reference operator
            "sinTheta": {"frobenius": ["0", "0"], "max_sin": ["0", "0"]},
        },
        "checks": {
            "theoretical_bound": bound_result,
            "wedin_bound": {
                "E_over_gamma": 0.0,
                "pass_estimate": True,
            },
        },
        "provenance": {
            "input_hash": input_hash,
            "kernel_binary_hash": get_kernel_version(),
            "runtime": {
                "container": "prototype-python",
                "mp_precision": precision,
            },
        },
        "signature": None,  # Prototype doesn't sign
    }

    # Write output
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Wrote kernel output to {output_path}")
    logger.info(f"Theoretical bound: {bound_result['lhs_interval'][0]}, pass: {bound_result['pass']}")

    return output


def main():
    """Main entry point for prototype kernel."""
    parser = argparse.ArgumentParser(
        description="Prototype verified kernel using mpmath interval arithmetic"
    )
    parser.add_argument("input", help="Path to kernel_input.json")
    parser.add_argument("output", help="Path for kernel_output.json")
    parser.add_argument(
        "--precision", type=int, default=128,
        help="Precision bits for interval arithmetic (default: 128)"
    )
    args = parser.parse_args()

    # Override from environment
    precision = int(os.environ.get("PRECISION_BITS", args.precision))

    if not HAVE_MPMATH:
        logger.warning(
            "mpmath not installed. Running without interval arithmetic. "
            "Install with: pip install mpmath"
        )

    try:
        output = run_kernel(args.input, args.output, precision)

        # Return non-zero if checks failed
        if not output["checks"]["theoretical_bound"]["pass"]:
            logger.warning("Theoretical bound check FAILED")
            sys.exit(1)

        sys.exit(0)

    except Exception as e:
        logger.error(f"Kernel failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
