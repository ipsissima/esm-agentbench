"""Validate runtime theoretical bounds against Coq-provided constants.

This script computes spectral certificates on synthetic datasets and verifies
that the runtime ``theoretical_bound`` is conservative, i.e.:

    theoretical_bound >= C_tail * tail_energy + C_res * residual

where C_tail and C_res are exported from the UELAT/Coq development.

Mathematical Basis:
- Uses SVD for spectral analysis (Wedin's Theorem for stability)
- No heuristic penalties; only rigorous bound formula
- Constants have proven upper bounds (C_res <= 2, C_tail <= 2)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from certificates import uelat_bridge
from certificates.make_certificate import compute_certificate

Array = np.ndarray
Generator = Callable[[int, int, np.random.Generator], Array]


def generate_ar1(T: int, d: int, rng: np.random.Generator) -> Array:
    """Generate an AR(1) process with stable coefficient."""
    rho = 0.8
    noise = rng.standard_normal((T, d))
    X = np.zeros((T, d))
    for t in range(1, T):
        X[t] = rho * X[t - 1] + 0.1 * noise[t]
    return X


def generate_affine(T: int, d: int, rng: np.random.Generator) -> Array:
    """Generate an affine-linear trajectory with small drift."""
    A = rng.standard_normal((d, d))
    A = A / max(np.linalg.norm(A, ord=2), 1.0) * 0.7
    b = rng.standard_normal(d) * 0.05
    X = np.zeros((T, d))
    for t in range(1, T):
        X[t] = A @ X[t - 1] + b
    return X


def generate_lowrank_noise(T: int, d: int, rng: np.random.Generator) -> Array:
    """Generate approximately low-rank data with additive noise."""
    r_true = max(1, d // 3)
    U = rng.standard_normal((d, r_true))
    V = rng.standard_normal((T, r_true))
    signal = V @ U.T
    noise = rng.standard_normal((T, d)) * 0.05
    return signal + noise


def _select_generators(names: Iterable[str]) -> Dict[str, Generator]:
    mapping = {
        "ar1": generate_ar1,
        "affine": generate_affine,
        "lowrank": generate_lowrank_noise,
    }
    selected = {}
    for name in names:
        if name == "all":
            return mapping
        if name not in mapping:
            raise ValueError(f"Unknown dataset '{name}'. Choose from {sorted(mapping)} or 'all'.")
        selected[name] = mapping[name]
    return selected


def _format_row(row: Tuple) -> str:
    return " | ".join(f"{v}" for v in row)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constants", type=str, default=None, help="Path to uelat_constants.json")
    parser.add_argument("--datasets", type=str, default="all", help="Comma-separated subset of ar1,affine,lowrank,all")
    parser.add_argument("--T", type=int, default=40, help="Number of timesteps")
    parser.add_argument("--d", type=int, default=8, help="Embedding dimension")
    parser.add_argument("--r-values", type=str, default="1,2,4", help="Comma-separated PCA ranks to test")
    parser.add_argument("--trials", type=int, default=3, help="Trials per dataset/r setting")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")

    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)

    # Load constants - use auto-load if no path specified
    if args.constants:
        constants = uelat_bridge.load_constants_from_json(args.constants, strict=False)
    else:
        constants = uelat_bridge.auto_load_constants(strict=False)

    c_tail = uelat_bridge.get_constant("C_tail")
    c_res = uelat_bridge.get_constant("C_res")

    print(f"Loaded constants: C_tail={c_tail}, C_res={c_res}")
    print()

    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]
    generators = _select_generators(dataset_names)

    r_values = [int(x) for x in args.r_values.split(",") if x]

    header = (
        "dataset",
        "r",
        "trial",
        "residual",
        "tail_energy",
        "theoretical_bound",
        "guaranteed_bound",
        "OK",
    )
    print(_format_row(header))
    failures: List[str] = []

    for dataset_name, generator in generators.items():
        for r in r_values:
            for trial in range(args.trials):
                seed = rng.integers(0, 2**32 - 1)
                trial_rng = np.random.default_rng(int(seed))
                X = generator(args.T, args.d, trial_rng)
                cert = compute_certificate(X, r=r)

                # Use tail_energy (new name) instead of pca_tail_estimate
                tail_energy = cert.get("tail_energy", cert.get("pca_tail_estimate", 0.0))
                guaranteed_bound = c_tail * tail_energy + c_res * cert["residual"]
                ok = cert["theoretical_bound"] >= guaranteed_bound - 1e-9

                row = (
                    dataset_name,
                    r,
                    trial,
                    f"{cert['residual']:.4f}",
                    f"{tail_energy:.4f}",
                    f"{cert['theoretical_bound']:.4f}",
                    f"{guaranteed_bound:.4f}",
                    "OK" if ok else "FAIL",
                )
                print(_format_row(row))
                if not ok:
                    failures.append(
                        json.dumps(
                            {
                                "dataset": dataset_name,
                                "r": r,
                                "trial": trial,
                                "certificate": {k: v for k, v in cert.items() if not isinstance(v, np.ndarray)},
                                "constants": constants,
                                "guaranteed_bound": guaranteed_bound,
                            },
                            indent=2,
                        )
                    )

    if failures:
        print("\nDiagnostics for failing trials:")
        for block in failures:
            print(block)
        return 1

    print("\nAll trials passed!")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
