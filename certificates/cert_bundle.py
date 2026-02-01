"""Certificate bundle creation and verification.

This module creates and verifies signed certificate bundles containing:
- trace.json: Original trace data
- kernel_input.json: Input to verified kernel
- kernel_output.json: Output from verified kernel
- certificate.json: Spectral certificate
- metadata.json: Bundle metadata and provenance
- signature.asc: GPG signature of the bundle

Signing requires GPG to be installed and configured with a signing key.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BundleError(Exception):
    """Raised when bundle operations fail."""
    pass


def compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """Compute hash of a file.

    Parameters
    ----------
    file_path : str
        Path to file.
    algorithm : str
        Hash algorithm (default: sha256).

    Returns
    -------
    str
        Hexadecimal hash string.
    """
    h = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def create_metadata(
    files: Dict[str, str],
    embedder_id: Optional[str] = None,
    kernel_mode: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create metadata for a certificate bundle.

    Parameters
    ----------
    files : Dict[str, str]
        Mapping of bundle file names to source paths.
    embedder_id : Optional[str]
        Embedding model identifier.
    kernel_mode : Optional[str]
        Kernel mode used (prototype, arb, mpfi).
    extra : Optional[Dict[str, Any]]
        Additional metadata to include.

    Returns
    -------
    Dict[str, Any]
        Metadata dictionary.
    """
    metadata = {
        "schema_version": "1.0",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "embedder_id": embedder_id or "unknown",
        "kernel_mode": kernel_mode or "unknown",
        "files": {},
    }

    # Add file hashes
    for name, path in files.items():
        if os.path.exists(path):
            metadata["files"][name] = {
                "sha256": compute_file_hash(path),
                "size_bytes": os.path.getsize(path),
            }

    # Add extra metadata
    if extra:
        metadata.update(extra)

    return metadata


def create_bundle(
    bundle_dir: str,
    trace_path: Optional[str] = None,
    kernel_input_path: Optional[str] = None,
    kernel_output_path: Optional[str] = None,
    certificate_path: Optional[str] = None,
    embedder_id: Optional[str] = None,
    kernel_mode: Optional[str] = None,
    gpg_key: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a signed certificate bundle.

    Creates a bundle directory containing all certificate artifacts
    and optionally signs it with GPG.

    Parameters
    ----------
    bundle_dir : str
        Directory to create bundle in.
    trace_path : Optional[str]
        Path to trace.json file.
    kernel_input_path : Optional[str]
        Path to kernel_input.json file.
    kernel_output_path : Optional[str]
        Path to kernel_output.json file.
    certificate_path : Optional[str]
        Path to certificate.json file.
    embedder_id : Optional[str]
        Embedding model identifier.
    kernel_mode : Optional[str]
        Kernel mode used.
    gpg_key : Optional[str]
        GPG key ID for signing. If None, no signature is created.
    extra_metadata : Optional[Dict[str, Any]]
        Additional metadata to include.

    Returns
    -------
    str
        Path to created bundle directory.

    Raises
    ------
    BundleError
        If bundle creation fails.
    """
    bundle_dir = os.path.abspath(bundle_dir)
    os.makedirs(bundle_dir, exist_ok=True)

    files: Dict[str, str] = {}

    # Copy files to bundle
    def copy_file(src: Optional[str], dest_name: str) -> None:
        if src and os.path.exists(src):
            dest = os.path.join(bundle_dir, dest_name)
            shutil.copy2(src, dest)
            files[dest_name] = src

    copy_file(trace_path, "trace.json")
    copy_file(kernel_input_path, "kernel_input.json")
    copy_file(kernel_output_path, "kernel_output.json")
    copy_file(certificate_path, "certificate.json")

    # Create metadata
    metadata = create_metadata(
        files={k: os.path.join(bundle_dir, k) for k in files},
        embedder_id=embedder_id,
        kernel_mode=kernel_mode,
        extra=extra_metadata,
    )

    # Write metadata
    metadata_path = os.path.join(bundle_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Sign if GPG key provided
    if gpg_key:
        try:
            sign_bundle(bundle_dir, gpg_key)
        except BundleError as e:
            logger.warning(f"Failed to sign bundle: {e}")

    logger.info(f"Created bundle at {bundle_dir}")
    return bundle_dir


def sign_bundle(bundle_dir: str, gpg_key: str) -> str:
    """Sign a certificate bundle with GPG.

    Creates a detached signature of the metadata.json file.

    Parameters
    ----------
    bundle_dir : str
        Path to bundle directory.
    gpg_key : str
        GPG key ID for signing.

    Returns
    -------
    str
        Path to signature file.

    Raises
    ------
    BundleError
        If signing fails.
    """
    if not shutil.which("gpg"):
        raise BundleError("GPG not found. Install GPG to sign bundles.")

    metadata_path = os.path.join(bundle_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise BundleError(f"metadata.json not found in bundle: {bundle_dir}")

    signature_path = os.path.join(bundle_dir, "signature.asc")

    cmd = [
        "gpg",
        "--armor",
        "--detach-sign",
        "--default-key", gpg_key,
        "--output", signature_path,
        metadata_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise BundleError(f"GPG signing failed: {result.stderr}")

        logger.info(f"Signed bundle with key {gpg_key}")
        return signature_path

    except subprocess.TimeoutExpired:
        raise BundleError("GPG signing timed out")


def verify_bundle(bundle_dir: str) -> Dict[str, Any]:
    """Verify a certificate bundle.

    Checks:
    1. All expected files are present
    2. File hashes match metadata
    3. GPG signature is valid (if present)

    Parameters
    ----------
    bundle_dir : str
        Path to bundle directory.

    Returns
    -------
    Dict[str, Any]
        Verification results including:
        - valid: True if all checks pass
        - checks: Dict of individual check results
        - errors: List of error messages

    Raises
    ------
    BundleError
        If bundle is missing or cannot be read.
    """
    bundle_dir = os.path.abspath(bundle_dir)
    if not os.path.isdir(bundle_dir):
        raise BundleError(f"Bundle directory not found: {bundle_dir}")

    results = {
        "valid": True,
        "checks": {},
        "errors": [],
    }

    # Check metadata exists
    metadata_path = os.path.join(bundle_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        results["valid"] = False
        results["errors"].append("metadata.json not found")
        return results

    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except json.JSONDecodeError as e:
        results["valid"] = False
        results["errors"].append(f"Invalid metadata.json: {e}")
        return results

    # Check required files
    for filename in ["kernel_output.json", "certificate.json"]:
        file_path = os.path.join(bundle_dir, filename)
        if os.path.exists(file_path):
            results["checks"][f"{filename}_present"] = True
        else:
            results["checks"][f"{filename}_present"] = False
            results["errors"].append(f"Missing required file: {filename}")
            results["valid"] = False

    # Verify file hashes
    for filename, file_meta in metadata.get("files", {}).items():
        file_path = os.path.join(bundle_dir, filename)
        if not os.path.exists(file_path):
            results["checks"][f"{filename}_hash"] = False
            results["errors"].append(f"File missing for hash check: {filename}")
            results["valid"] = False
            continue

        expected_hash = file_meta.get("sha256")
        if expected_hash:
            actual_hash = compute_file_hash(file_path)
            if actual_hash == expected_hash:
                results["checks"][f"{filename}_hash"] = True
            else:
                results["checks"][f"{filename}_hash"] = False
                results["errors"].append(
                    f"Hash mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
                )
                results["valid"] = False

    # Check GPG signature if present
    signature_path = os.path.join(bundle_dir, "signature.asc")
    if os.path.exists(signature_path):
        if shutil.which("gpg"):
            try:
                cmd = ["gpg", "--verify", signature_path, metadata_path]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode == 0:
                    results["checks"]["signature"] = True
                else:
                    results["checks"]["signature"] = False
                    results["errors"].append(f"Signature verification failed: {result.stderr}")
                    results["valid"] = False
            except subprocess.TimeoutExpired:
                results["checks"]["signature"] = False
                results["errors"].append("Signature verification timed out")
                results["valid"] = False
        else:
            results["checks"]["signature"] = None
            logger.warning("GPG not available for signature verification")
    else:
        results["checks"]["signature"] = None  # No signature to verify

    return results


def create_bundle_archive(
    bundle_dir: str,
    output_path: Optional[str] = None,
    gpg_key: Optional[str] = None,
) -> str:
    """Create a compressed archive of a certificate bundle.

    Parameters
    ----------
    bundle_dir : str
        Path to bundle directory.
    output_path : Optional[str]
        Path for output archive. If None, uses bundle_dir.tar.gz.
    gpg_key : Optional[str]
        GPG key ID for signing the archive. If None, no signature.

    Returns
    -------
    str
        Path to created archive.
    """
    bundle_dir = os.path.abspath(bundle_dir)
    if output_path is None:
        output_path = f"{bundle_dir}.tar.gz"

    # Create tar.gz archive
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(bundle_dir, arcname=os.path.basename(bundle_dir))

    logger.info(f"Created archive: {output_path}")

    # Sign archive if GPG key provided
    if gpg_key and shutil.which("gpg"):
        sig_path = f"{output_path}.asc"
        cmd = [
            "gpg",
            "--armor",
            "--detach-sign",
            "--default-key", gpg_key,
            "--output", sig_path,
            output_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=60, check=True)
            logger.info(f"Signed archive: {sig_path}")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            logger.warning(f"Failed to sign archive: {e}")

    return output_path


def extract_bundle_archive(archive_path: str, output_dir: str) -> str:
    """Extract a certificate bundle archive.

    Parameters
    ----------
    archive_path : str
        Path to bundle archive (tar.gz).
    output_dir : str
        Directory to extract to.

    Returns
    -------
    str
        Path to extracted bundle directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(output_dir)

    # Find the bundle directory
    entries = os.listdir(output_dir)
    if len(entries) == 1 and os.path.isdir(os.path.join(output_dir, entries[0])):
        return os.path.join(output_dir, entries[0])
    return output_dir


__all__ = [
    "create_bundle",
    "sign_bundle",
    "verify_bundle",
    "create_bundle_archive",
    "extract_bundle_archive",
    "create_metadata",
    "compute_file_hash",
    "BundleError",
]
