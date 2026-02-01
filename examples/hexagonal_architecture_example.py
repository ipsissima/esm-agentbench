"""Example application demonstrating hexagonal architecture.

This example shows how to use the service layer to generate certificates
with different configurations (basic, with verification, with signing).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from services import CertificateService, create_kernel_adapter, create_signer, create_storage


def load_trace(trace_path: str) -> Dict[str, Any]:
    """Load trace from JSON file."""
    with open(trace_path, "r") as handle:
        return json.load(handle)


def example_basic(trace_path: str, output_path: str | None = None) -> Dict[str, Any]:
    """Example: Basic certificate generation without verification.
    
    This uses the service layer with no adapters configured.
    """
    print("=== Example 1: Basic Certificate Generation ===")
    
    # Create service with no adapters
    service = CertificateService()
    
    # Load trace and generate certificate
    trace = load_trace(trace_path)
    certificate = service.generate_certificate(trace, rank=10)
    
    print(f"✓ Certificate generated")
    print(f"  Residual: {certificate.get('residual', 'N/A')}")
    print(f"  Bound: {certificate.get('theoretical_bound', 'N/A')}")
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump(certificate, f, indent=2)
        print(f"✓ Certificate saved to {output_path}")
    
    return certificate


def example_verified(trace_path: str, output_path: str | None = None) -> Dict[str, Any]:
    """Example: Certificate generation with kernel verification.
    
    This uses the kernel adapter for verified computation.
    """
    print("\n=== Example 2: Verified Certificate Generation ===")
    
    # Create service with kernel adapter
    service = CertificateService(
        kernel=create_kernel_adapter()
    )
    
    # Load trace and generate certificate with verification
    trace = load_trace(trace_path)
    certificate = service.generate_certificate(
        trace,
        rank=10,
        verify_with_kernel=True,
    )
    
    print(f"✓ Verified certificate generated")
    print(f"  Residual: {certificate.get('residual', 'N/A')}")
    print(f"  Bound: {certificate.get('theoretical_bound', 'N/A')}")
    
    if "kernel_output" in certificate:
        print(f"  Kernel verification: {certificate['kernel_output'].get('status', 'N/A')}")
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump(certificate, f, indent=2)
        print(f"✓ Verified certificate saved to {output_path}")
    
    return certificate


def example_signed(
    trace_path: str,
    signer_id: str,
    gpg_key: str,
    output_path: str | None = None,
) -> Dict[str, Any]:
    """Example: Certificate generation with signing.
    
    This uses the signer adapter to cryptographically sign the certificate.
    """
    print("\n=== Example 3: Signed Certificate Generation ===")
    
    # Create service with signer adapter
    service = CertificateService(
        signer=create_signer(signer_id, gpg_key)
    )
    
    # Load trace and generate signed certificate
    trace = load_trace(trace_path)
    certificate = service.generate_and_sign(trace, rank=10)
    
    print(f"✓ Signed certificate generated")
    print(f"  Residual: {certificate.get('residual', 'N/A')}")
    print(f"  Bound: {certificate.get('theoretical_bound', 'N/A')}")
    print(f"  Signer: {certificate.get('signature', {}).get('signer_id', 'N/A')}")
    
    if output_path:
        with open(output_path, "w") as f:
            json.dump(certificate, f, indent=2)
        print(f"✓ Signed certificate saved to {output_path}")
    
    return certificate


def example_full_featured(
    trace_path: str,
    signer_id: str,
    gpg_key: str,
    storage_dir: str,
) -> Dict[str, Any]:
    """Example: Full-featured certificate with verification, signing, and storage.
    
    This demonstrates using multiple adapters together.
    """
    print("\n=== Example 4: Full-Featured Certificate ===")
    
    # Create service with all adapters
    service = CertificateService(
        kernel=create_kernel_adapter(),
        signer=create_signer(signer_id, gpg_key),
        storage=create_storage(Path(storage_dir)),
    )
    
    # Load trace
    trace = load_trace(trace_path)
    
    # Generate certificate with verification
    certificate = service.generate_certificate(
        trace,
        rank=10,
        verify_with_kernel=True,
    )
    
    # Sign the certificate
    import json as json_lib
    certificate_bytes = json_lib.dumps(certificate, sort_keys=True).encode("utf-8")
    signature = service._signer.sign_bytes(certificate_bytes)
    certificate["signature"] = {
        "signer_id": signer_id,
        "signature_bytes": signature.hex(),
    }
    
    # Persist to storage
    storage_path = service._storage.save_trace(
        metadata=trace.get("metadata", {}),
        steps=trace.get("steps", []),
        outcome=certificate,
    )
    
    print(f"✓ Full-featured certificate generated")
    print(f"  Residual: {certificate.get('residual', 'N/A')}")
    print(f"  Bound: {certificate.get('theoretical_bound', 'N/A')}")
    print(f"  Verified: {('kernel_output' in certificate)}")
    print(f"  Signed: {('signature' in certificate)}")
    print(f"  Stored: {storage_path}")
    
    return certificate


def main():
    """Run example scenarios."""
    parser = argparse.ArgumentParser(
        description="Example application for hexagonal architecture"
    )
    parser.add_argument(
        "trace_path",
        help="Path to trace JSON file",
    )
    parser.add_argument(
        "--example",
        choices=["basic", "verified", "signed", "full"],
        default="basic",
        help="Which example to run",
    )
    parser.add_argument(
        "--output",
        help="Output path for certificate JSON",
    )
    parser.add_argument(
        "--signer-id",
        default="example-signer",
        help="Signer identifier (for signed/full examples)",
    )
    parser.add_argument(
        "--gpg-key",
        default="example@example.com",
        help="GPG key ID (for signed/full examples)",
    )
    parser.add_argument(
        "--storage-dir",
        default="/tmp/traces",
        help="Storage directory (for full example)",
    )
    
    args = parser.parse_args()
    
    try:
        if args.example == "basic":
            example_basic(args.trace_path, args.output)
        elif args.example == "verified":
            example_verified(args.trace_path, args.output)
        elif args.example == "signed":
            example_signed(
                args.trace_path,
                args.signer_id,
                args.gpg_key,
                args.output,
            )
        elif args.example == "full":
            example_full_featured(
                args.trace_path,
                args.signer_id,
                args.gpg_key,
                args.storage_dir,
            )
        
        print("\n✓ Example completed successfully")
    
    except Exception as exc:
        print(f"\n✗ Example failed: {exc}")
        raise


if __name__ == "__main__":
    main()
