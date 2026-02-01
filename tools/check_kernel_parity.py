#!/usr/bin/env python3
"""Compare prototype kernel output with ARB kernel output for interval parity."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, List, Tuple

import numpy as np

from certificates.make_certificate import export_kernel_input


def _parse_interval(interval: Iterable[str]) -> Tuple[float, float]:
    low, high = interval
    return float(low), float(high)


def _intervals_intersect(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return a[1] >= b[0] and b[1] >= a[0]


def _compare_interval_lists(label: str, a_list: List[List[str]], b_list: List[List[str]]) -> None:
    if len(a_list) != len(b_list):
        raise AssertionError(f"{label}: length mismatch {len(a_list)} vs {len(b_list)}")
    for idx, (a_interval, b_interval) in enumerate(zip(a_list, b_list)):
        a_pair = _parse_interval(a_interval)
        b_pair = _parse_interval(b_interval)
        if not _intervals_intersect(a_pair, b_pair):
            raise AssertionError(f"{label}[{idx}] intervals do not intersect: {a_pair} vs {b_pair}")


def _compare_interval(label: str, a_interval: List[str], b_interval: List[str]) -> None:
    a_pair = _parse_interval(a_interval)
    b_pair = _parse_interval(b_interval)
    if not _intervals_intersect(a_pair, b_pair):
        raise AssertionError(f"{label} intervals do not intersect: {a_pair} vs {b_pair}")


def _load_output(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _run_prototype(kernel_input: str, kernel_output: str, precision: int) -> None:
    cmd = [
        "python",
        os.path.join("kernel", "prototype", "prototype_kernel.py"),
        kernel_input,
        kernel_output,
        "--precision",
        str(precision),
    ]
    subprocess.run(cmd, check=True)


def _run_arb(
    kernel_input: str,
    kernel_output: str,
    precision: int,
    docker_image: str,
    arb_command: str | None,
) -> None:
    if arb_command:
        cmd = arb_command.format(
            kernel_input=kernel_input,
            kernel_output=kernel_output,
            precision=precision,
        )
        subprocess.run(cmd, shell=True, check=True)
        return

    if not shutil.which("docker"):
        raise RuntimeError("docker is required to run the ARB kernel container")

    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{kernel_input}:/data/kernel_input.json:ro",
        "-v",
        f"{kernel_output}:/data/kernel_output.json:rw",
        "-e",
        f"PRECISION_BITS={precision}",
        docker_image,
        "/data/kernel_input.json",
        "/data/kernel_output.json",
    ]
    subprocess.run(cmd, check=True)


def _compare_outputs(proto: Any, arb: Any) -> None:
    proto_comp = proto["computed"]
    arb_comp = arb["computed"]

    _compare_interval_lists("sigma", proto_comp["sigma"], arb_comp["sigma"])
    _compare_interval("gamma", proto_comp["gamma"], arb_comp["gamma"])
    _compare_interval("tail_energy", proto_comp["tail_energy"], arb_comp["tail_energy"])
    _compare_interval("pca_explained", proto_comp["pca_explained"], arb_comp["pca_explained"])

    proto_koop = proto_comp["koopman"]
    arb_koop = arb_comp["koopman"]
    _compare_interval_lists("koopman_sigma", proto_koop["koopman_sigma"], arb_koop["koopman_sigma"])
    _compare_interval(
        "koopman_singular_gap",
        proto_koop["koopman_singular_gap"],
        arb_koop["koopman_singular_gap"],
    )

    proto_res = proto_comp["residuals"]
    arb_res = arb_comp["residuals"]
    _compare_interval("oos_residual", proto_res["oos_residual"], arb_res["oos_residual"])
    _compare_interval("insample_residual", proto_res["insample_residual"], arb_res["insample_residual"])

    if proto_res.get("r_t_intervals") and arb_res.get("r_t_intervals"):
        _compare_interval_lists("r_t_intervals", proto_res["r_t_intervals"], arb_res["r_t_intervals"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare prototype and ARB kernel outputs.")
    parser.add_argument("--samples", type=int, default=3, help="Number of random samples to compare.")
    parser.add_argument("--rank", type=int, default=5, help="Target rank for kernel input.")
    parser.add_argument("--precision", type=int, default=128, help="Precision bits for kernels.")
    parser.add_argument(
        "--docker-image",
        default="ipsissima/kernel:parity",
        help="Docker image tag for ARB kernel.",
    )
    parser.add_argument(
        "--arb-command",
        default=None,
        help="Optional command template to run the ARB kernel.",
    )
    args = parser.parse_args()

    for idx in range(args.samples):
        X = np.random.randn(12, 8)
        X_aug = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        trace_id = f"parity-{idx}"

        with tempfile.TemporaryDirectory(prefix="kernel_parity_") as tmpdir:
            kernel_input = os.path.join(tmpdir, "kernel_input.json")
            proto_output = os.path.join(tmpdir, "kernel_output_proto.json")
            arb_output = os.path.join(tmpdir, "kernel_output_arb.json")

            export_kernel_input(
                X_aug=X_aug,
                trace_id=trace_id,
                output_path=kernel_input,
                rank=args.rank,
                precision_bits=args.precision,
                kernel_mode="prototype",
            )

            _run_prototype(kernel_input, proto_output, args.precision)
            _run_arb(kernel_input, arb_output, args.precision, args.docker_image, args.arb_command)

            proto = _load_output(proto_output)
            arb = _load_output(arb_output)

            _compare_outputs(proto, arb)

    print("Kernel parity checks passed.")


if __name__ == "__main__":
    main()
