"""Application entrypoint for certificate generation."""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from adapters.kernel_client import KernelClientAdapter
from core.certificate import compute_certificate_from_trace


def load_trace(trace_path: str) -> Dict[str, Any]:
    with open(trace_path, "r") as handle:
        return json.load(handle)


def run(
    trace_path: str,
    *,
    output_path: str | None = None,
    rank: int = 10,
    psi_mode: str = "embedding",
    verify_with_kernel: bool = False,
    kernel_mode: str = "prototype",
    precision_bits: int = 128,
) -> Dict[str, Any]:
    trace = load_trace(trace_path)
    kernel = KernelClientAdapter() if verify_with_kernel else None
    certificate = compute_certificate_from_trace(
        trace,
        psi_mode=psi_mode,
        rank=rank,
        kernel=kernel,
        kernel_mode=kernel_mode,
        precision_bits=precision_bits,
    )

    if output_path:
        with open(output_path, "w") as handle:
            json.dump(certificate, handle, indent=2)

    return certificate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate spectral certificates")
    parser.add_argument("trace_path", help="Path to trace JSON")
    parser.add_argument("--output", help="Write certificate JSON to file")
    parser.add_argument("--rank", type=int, default=10, help="SVD rank")
    parser.add_argument("--psi-mode", default="embedding", help="embedding or residual_stream")
    parser.add_argument("--verify-with-kernel", action="store_true", help="Run verified kernel")
    parser.add_argument("--kernel-mode", default="prototype", help="prototype/arb/mpfi")
    parser.add_argument("--precision-bits", type=int, default=128)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(
        args.trace_path,
        output_path=args.output,
        rank=args.rank,
        psi_mode=args.psi_mode,
        verify_with_kernel=args.verify_with_kernel,
        kernel_mode=args.kernel_mode,
        precision_bits=args.precision_bits,
    )


if __name__ == "__main__":
    main()
