#!/usr/bin/env python3
"""Compare prototype and ARB kernels for parity on random inputs."""
from __future__ import annotations

import argparse
import json
import tempfile
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from certificates.kernel_client import KernelClientError, run_kernel
from certificates.make_certificate import export_kernel_input


def parse_interval(interval: Iterable[str]) -> Tuple[float, float]:
    values = [float(x) for x in interval]
    if len(values) != 2:
        raise ValueError(f"Expected interval of length 2, got {interval}")
    low, high = values
    if low > high:
        low, high = high, low
    return low, high


def intervals_intersect(a: Iterable[str], b: Iterable[str]) -> bool:
    a_low, a_high = parse_interval(a)
    b_low, b_high = parse_interval(b)
    return max(a_low, b_low) <= min(a_high, b_high)


def compare_intervals(
    prototype: Dict[str, Any],
    arb: Dict[str, Any],
) -> Tuple[bool, list[str]]:
    errors: list[str] = []

    def compare_field(path: str, proto_interval: Iterable[str], arb_interval: Iterable[str]) -> None:
        if not intervals_intersect(proto_interval, arb_interval):
            errors.append(f"Interval mismatch at {path}: {proto_interval} vs {arb_interval}")

    proto_computed = prototype.get("computed", {})
    arb_computed = arb.get("computed", {})

    for key in ("gamma", "tail_energy", "pca_explained"):
        if key in proto_computed and key in arb_computed:
            compare_field(f"computed.{key}", proto_computed[key], arb_computed[key])
        else:
            errors.append(f"Missing computed.{key} in output")

    proto_sigma = proto_computed.get("sigma", [])
    arb_sigma = arb_computed.get("sigma", [])
    if len(proto_sigma) != len(arb_sigma):
        errors.append(f"Sigma length mismatch: {len(proto_sigma)} vs {len(arb_sigma)}")
    else:
        for idx, (proto_interval, arb_interval) in enumerate(zip(proto_sigma, arb_sigma)):
            compare_field(f"computed.sigma[{idx}]", proto_interval, arb_interval)

    proto_residuals = proto_computed.get("residuals", {})
    arb_residuals = arb_computed.get("residuals", {})
    for key in ("insample_residual", "oos_residual"):
        if key in proto_residuals and key in arb_residuals:
            compare_field(f"computed.residuals.{key}", proto_residuals[key], arb_residuals[key])
        else:
            errors.append(f"Missing computed.residuals.{key} in output")

    return not errors, errors


def run_parity_check(
    *,
    n: int,
    precision_bits: int,
    rank: int,
    seed: int,
    require_arb: bool,
) -> None:
    rng = np.random.default_rng(seed)
    failures = 0

    for idx in range(n):
        rows = rng.integers(8, 16)
        cols = rng.integers(4, 10)
        X = rng.standard_normal((rows, cols))
        X_aug = np.concatenate([X, np.ones((rows, 1))], axis=1)

        with tempfile.TemporaryDirectory(prefix="esm_kernel_parity_") as tmpdir:
            kernel_input_path = f"{tmpdir}/kernel_input.json"
            export_kernel_input(
                X_aug,
                trace_id=f"parity-{idx}",
                output_path=kernel_input_path,
                rank=rank,
                precision_bits=precision_bits,
            )

            prototype_output = run_kernel(
                kernel_input_path,
                precision_bits=precision_bits,
                mode="prototype",
            )

            try:
                arb_output = run_kernel(
                    kernel_input_path,
                    precision_bits=precision_bits,
                    mode="arb",
                )
            except KernelClientError as exc:
                if require_arb:
                    raise
                print(f"Skipping ARB run: {exc}")
                return

        ok, errors = compare_intervals(prototype_output, arb_output)
        if not ok:
            failures += 1
            print(f"Parity mismatch for sample {idx}:")
            for error in errors:
                print(f"  - {error}")

    if failures:
        raise SystemExit(f"Parity check failed for {failures} of {n} samples.")

    print(f"Parity check passed for {n} samples.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check parity between kernel modes.")
    parser.add_argument("--n", type=int, default=10, help="Number of random samples")
    parser.add_argument("--precision", type=int, default=256, help="Precision bits")
    parser.add_argument("--rank", type=int, default=8, help="Rank for SVD")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--allow-missing-arb",
        action="store_true",
        help="Skip ARB parity if docker kernel is unavailable",
    )
    args = parser.parse_args()

    run_parity_check(
        n=args.n,
        precision_bits=args.precision,
        rank=args.rank,
        seed=args.seed,
        require_arb=not args.allow_missing_arb,
    )


if __name__ == "__main__":
    main()
