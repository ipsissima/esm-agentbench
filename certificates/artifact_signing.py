"""Artifact Signing and Verification System.

Provides cryptographic signing and verification for:
- Verified kernel binaries (.so files)
- Trace certificates
- Model attestations

Uses Ed25519 signatures via PyNaCl for signing. Verification can use
either PyNaCl (preferred) or pure Python (fallback).

Security Properties:
- Signatures are detached (stored separately from artifacts)
- Public keys can be pinned in CI for verification
- Supports multiple signers (e.g., CI bot, release manager)

Usage:
    # Sign an artifact
    signer = ArtifactSigner.from_key_file("private.key")
    signer.sign_file("kernel_verified.so", "kernel_verified.so.sig")

    # Verify an artifact
    verifier = ArtifactVerifier(public_key=TRUSTED_PUBLIC_KEY)
    if verifier.verify_file("kernel_verified.so", "kernel_verified.so.sig"):
        print("Signature valid!")
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import struct
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Signature file format version
SIGNATURE_VERSION = 1

# Trusted public keys for CI verification (Ed25519 public keys, base64-encoded)
# These should be pinned in CI and not changed without proper key rotation
TRUSTED_CI_PUBLIC_KEYS: List[str] = [
    # CI bot key (placeholder - replace with actual key)
    # "base64-encoded-ed25519-public-key-here"
]


class SigningError(RuntimeError):
    """Raised when signing operation fails."""


class VerificationError(RuntimeError):
    """Raised when verification operation fails."""


@dataclass
class ArtifactManifest:
    """Manifest describing a signed artifact.

    Contains metadata about the artifact and cryptographic hashes
    for verification even without the signature.
    """
    version: int
    filename: str
    size: int
    sha256: str
    sha512: str
    timestamp: float
    signer_id: str
    extra_metadata: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, data: str) -> "ArtifactManifest":
        d = json.loads(data)
        return cls(**d)

    def canonical_bytes(self) -> bytes:
        """Get canonical bytes representation for signing.

        Uses sorted keys and no whitespace for deterministic output.
        """
        d = asdict(self)
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass
class SignedArtifact:
    """A signed artifact with manifest and signature."""
    manifest: ArtifactManifest
    signature: bytes  # Ed25519 signature over manifest.canonical_bytes()

    def to_file(self, path: Path) -> None:
        """Write signature file."""
        data = {
            "version": SIGNATURE_VERSION,
            "manifest": asdict(self.manifest),
            "signature": base64.b64encode(self.signature).decode("ascii"),
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def from_file(cls, path: Path) -> "SignedArtifact":
        """Read signature file."""
        data = json.loads(path.read_text())
        if data.get("version") != SIGNATURE_VERSION:
            raise VerificationError(
                f"Unsupported signature version: {data.get('version')}"
            )
        manifest = ArtifactManifest(**data["manifest"])
        signature = base64.b64decode(data["signature"])
        return cls(manifest=manifest, signature=signature)


def compute_file_hashes(filepath: Path) -> Tuple[str, str]:
    """Compute SHA-256 and SHA-512 hashes of a file.

    Parameters
    ----------
    filepath : Path
        Path to file to hash.

    Returns
    -------
    Tuple[str, str]
        (sha256_hex, sha512_hex)
    """
    sha256 = hashlib.sha256()
    sha512 = hashlib.sha512()

    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            sha256.update(chunk)
            sha512.update(chunk)

    return sha256.hexdigest(), sha512.hexdigest()


class ArtifactSigner:
    """Signs artifacts using Ed25519.

    Requires PyNaCl for signing (verification can use fallback).
    """

    def __init__(self, private_key: bytes, signer_id: str = "default"):
        """Initialize signer with Ed25519 private key.

        Parameters
        ----------
        private_key : bytes
            Ed25519 private key (32 bytes seed).
        signer_id : str
            Identifier for this signer (e.g., "ci-bot", "release-manager").
        """
        try:
            from nacl.signing import SigningKey
        except ImportError:
            raise SigningError(
                "PyNaCl required for signing. Install with: pip install pynacl"
            )

        self._key = SigningKey(private_key)
        self.signer_id = signer_id

    @classmethod
    def from_key_file(cls, path: Path, signer_id: str = "default") -> "ArtifactSigner":
        """Load signer from key file.

        Key file should contain base64-encoded 32-byte Ed25519 seed.
        """
        key_data = Path(path).read_text().strip()
        private_key = base64.b64decode(key_data)
        if len(private_key) != 32:
            raise SigningError(f"Invalid key length: {len(private_key)} (expected 32)")
        return cls(private_key, signer_id)

    @classmethod
    def generate(cls, signer_id: str = "default") -> Tuple["ArtifactSigner", str]:
        """Generate a new signing key pair.

        Returns
        -------
        Tuple[ArtifactSigner, str]
            (signer, base64_public_key)
        """
        try:
            from nacl.signing import SigningKey
        except ImportError:
            raise SigningError("PyNaCl required for key generation")

        key = SigningKey.generate()
        signer = cls(bytes(key), signer_id)
        public_key = base64.b64encode(bytes(key.verify_key)).decode("ascii")
        return signer, public_key

    def get_public_key(self) -> str:
        """Get base64-encoded public key."""
        return base64.b64encode(bytes(self._key.verify_key)).decode("ascii")

    def sign_file(
        self,
        filepath: Path,
        output_path: Optional[Path] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> SignedArtifact:
        """Sign a file and write signature.

        Parameters
        ----------
        filepath : Path
            Path to file to sign.
        output_path : Path, optional
            Path for signature file. Defaults to filepath + ".sig".
        extra_metadata : dict, optional
            Additional metadata to include in manifest.

        Returns
        -------
        SignedArtifact
            The signed artifact.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise SigningError(f"File not found: {filepath}")

        output_path = output_path or filepath.with_suffix(filepath.suffix + ".sig")

        # Compute hashes
        sha256, sha512 = compute_file_hashes(filepath)

        # Create manifest
        manifest = ArtifactManifest(
            version=SIGNATURE_VERSION,
            filename=filepath.name,
            size=filepath.stat().st_size,
            sha256=sha256,
            sha512=sha512,
            timestamp=time.time(),
            signer_id=self.signer_id,
            extra_metadata=extra_metadata or {},
        )

        # Sign manifest
        signed = self._key.sign(manifest.canonical_bytes())
        signature = signed.signature

        # Create signed artifact
        artifact = SignedArtifact(manifest=manifest, signature=signature)
        artifact.to_file(output_path)

        logger.info(f"Signed {filepath} -> {output_path}")
        return artifact

    def sign_bytes(self, data: bytes) -> bytes:
        """Sign arbitrary bytes.

        Parameters
        ----------
        data : bytes
            Data to sign.

        Returns
        -------
        bytes
            Ed25519 signature.
        """
        signed = self._key.sign(data)
        return signed.signature


class ArtifactVerifier:
    """Verifies artifact signatures.

    Can use PyNaCl (preferred) or pure Python (fallback).
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        trusted_keys: Optional[List[str]] = None,
    ):
        """Initialize verifier.

        Parameters
        ----------
        public_key : str, optional
            Single base64-encoded public key to trust.
        trusted_keys : List[str], optional
            List of trusted public keys. If None, uses TRUSTED_CI_PUBLIC_KEYS.
        """
        self._trusted_keys: List[bytes] = []

        # Add single key if provided
        if public_key:
            self._trusted_keys.append(base64.b64decode(public_key))

        # Add trusted keys
        if trusted_keys:
            for key in trusted_keys:
                self._trusted_keys.append(base64.b64decode(key))
        elif not public_key:
            # Use default CI keys if no keys specified
            for key in TRUSTED_CI_PUBLIC_KEYS:
                if key:  # Skip empty placeholders
                    self._trusted_keys.append(base64.b64decode(key))

        # Try to use PyNaCl for verification
        self._use_nacl = False
        try:
            from nacl.signing import VerifyKey
            self._use_nacl = True
        except ImportError:
            logger.debug("PyNaCl not available, using pure Python verification")

    def verify_file(
        self,
        filepath: Path,
        signature_path: Optional[Path] = None,
        check_hashes: bool = True,
    ) -> bool:
        """Verify a signed file.

        Parameters
        ----------
        filepath : Path
            Path to file to verify.
        signature_path : Path, optional
            Path to signature file. Defaults to filepath + ".sig".
        check_hashes : bool
            Whether to verify file hashes match manifest.

        Returns
        -------
        bool
            True if signature is valid.

        Raises
        ------
        VerificationError
            If verification fails.
        """
        filepath = Path(filepath)
        signature_path = signature_path or filepath.with_suffix(filepath.suffix + ".sig")

        if not filepath.exists():
            raise VerificationError(f"File not found: {filepath}")
        if not signature_path.exists():
            raise VerificationError(f"Signature file not found: {signature_path}")

        # Load signed artifact
        artifact = SignedArtifact.from_file(signature_path)

        # Verify hashes if requested
        if check_hashes:
            sha256, sha512 = compute_file_hashes(filepath)
            if sha256 != artifact.manifest.sha256:
                raise VerificationError(
                    f"SHA-256 mismatch: expected {artifact.manifest.sha256}, got {sha256}"
                )
            if sha512 != artifact.manifest.sha512:
                raise VerificationError(
                    f"SHA-512 mismatch: expected {artifact.manifest.sha512}, got {sha512}"
                )

            # Also check file size
            if filepath.stat().st_size != artifact.manifest.size:
                raise VerificationError(
                    f"Size mismatch: expected {artifact.manifest.size}, got {filepath.stat().st_size}"
                )

        # Verify signature against trusted keys
        manifest_bytes = artifact.manifest.canonical_bytes()

        for public_key in self._trusted_keys:
            try:
                if self._verify_signature(
                    manifest_bytes, artifact.signature, public_key
                ):
                    logger.info(f"Verified {filepath} with trusted key")
                    return True
            except Exception:
                continue

        raise VerificationError(
            f"Signature verification failed: no trusted key matched"
        )

    def _verify_signature(
        self, message: bytes, signature: bytes, public_key: bytes
    ) -> bool:
        """Verify Ed25519 signature.

        Returns True if the signature is valid. Prefer PyNaCl (fast),
        but raise VerificationError when PyNaCl is not available.

        Parameters
        ----------
        message : bytes
            Message that was signed.
        signature : bytes
            Ed25519 signature (64 bytes).
        public_key : bytes
            Ed25519 public key (32 bytes).

        Returns
        -------
        bool
            True if signature is valid.
        """
        if self._use_nacl:
            # Try to use PyNaCl's VerifyKey. Be tolerant of PyNaCl API
            # differences between versions.
            try:
                from nacl.signing import VerifyKey
            except Exception as e:
                logger.debug("PyNaCl VerifyKey import failed: %s", e)
                raise VerificationError(
                    "PyNaCl import failed during signature verification. "
                    f"Install with: pip install pynacl. Error: {e}"
                )

            # Prefer canonical BadSignatureError if available; otherwise fall back
            # to other known names or a generic Exception catch.
            _BadSigExc: type = Exception  # type: ignore
            try:
                from nacl.exceptions import BadSignatureError
                _BadSigExc = BadSignatureError
            except ImportError:
                try:
                    from nacl.exceptions import BadSignature  # type: ignore
                    _BadSigExc = BadSignature
                except ImportError:
                    try:
                        from nacl.signing import BadSignature  # type: ignore
                        _BadSigExc = BadSignature
                    except ImportError:
                        # Fall back to catching generic Exception
                        logger.debug(
                            "Could not import BadSignature/BadSignatureError; "
                            "will catch generic Exception for verification failures"
                        )

            try:
                verify_key = VerifyKey(public_key)
                # PyNaCl will raise a BadSignatureError (or variant) if verification fails.
                verify_key.verify(message, signature)
                return True
            except _BadSigExc:
                # Expected: signature invalid
                return False
            except Exception as exc:
                # Unexpected error (corrupt key, API mismatch, etc.)
                # Log at DEBUG so CI logs aren't noisy but we can diagnose.
                logger.debug(
                    "Unexpected exception during PyNaCl verification; treating as failure: %s",
                    exc,
                )
                return False
        else:
            # Pure Python Ed25519 verification is complex and error-prone
            # For security, we require PyNaCl for verification in production
            raise VerificationError(
                "PyNaCl required for signature verification. "
                "Install with: pip install pynacl"
            )

    def verify_hash_only(self, filepath: Path, expected_sha256: str) -> bool:
        """Verify file hash only (no signature check).

        Use this when signature verification is not possible but
        hash is known from a trusted source (e.g., CI artifact metadata).

        Parameters
        ----------
        filepath : Path
            Path to file to verify.
        expected_sha256 : str
            Expected SHA-256 hash (hex string).

        Returns
        -------
        bool
            True if hash matches.
        """
        sha256, _ = compute_file_hashes(filepath)
        return sha256 == expected_sha256


def verify_kernel_artifact(
    kernel_path: Path,
    signature_path: Optional[Path] = None,
    trusted_key: Optional[str] = None,
    allow_unsigned: bool = False,
) -> bool:
    """Verify a kernel artifact before loading.

    This is the main entry point for kernel verification.

    Parameters
    ----------
    kernel_path : Path
        Path to kernel shared library.
    signature_path : Path, optional
        Path to signature file.
    trusted_key : str, optional
        Base64-encoded public key to trust.
    allow_unsigned : bool
        If True, allow unsigned kernels (for development).

    Returns
    -------
    bool
        True if verification passes.

    Raises
    ------
    VerificationError
        If verification fails and allow_unsigned is False.
    """
    kernel_path = Path(kernel_path)
    signature_path = signature_path or kernel_path.with_suffix(kernel_path.suffix + ".sig")

    # Check if signature exists
    if not signature_path.exists():
        if allow_unsigned:
            logger.warning(f"No signature for {kernel_path}, allowing unsigned")
            return True
        raise VerificationError(f"No signature file found: {signature_path}")

    # Verify signature
    verifier = ArtifactVerifier(public_key=trusted_key)
    return verifier.verify_file(kernel_path, signature_path)


__all__ = [
    "ArtifactSigner",
    "ArtifactVerifier",
    "ArtifactManifest",
    "SignedArtifact",
    "SigningError",
    "VerificationError",
    "compute_file_hashes",
    "verify_kernel_artifact",
]
