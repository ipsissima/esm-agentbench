"""Certificate service for coordinating certificate generation.

This service provides a high-level API for generating spectral certificates
from traces, coordinating between the domain logic and infrastructure adapters.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from core.certificate import compute_certificate_from_trace
from ports.embedder import EmbedderPort
from ports.kernel import KernelClientPort
from ports.signer import SignerPort
from ports.storage import TraceStoragePort


class CertificateService:
    """Service for generating and managing spectral certificates.
    
    This service coordinates certificate generation by orchestrating
    domain logic with infrastructure adapters (ports).
    """

    def __init__(
        self,
        *,
        embedder: Optional[EmbedderPort] = None,
        kernel: Optional[KernelClientPort] = None,
        signer: Optional[SignerPort] = None,
        storage: Optional[TraceStoragePort] = None,
    ) -> None:
        """Initialize certificate service with optional adapters.
        
        Parameters
        ----------
        embedder : EmbedderPort, optional
            Embedder adapter for computing embeddings.
        kernel : KernelClientPort, optional
            Kernel adapter for verified computation.
        signer : SignerPort, optional
            Signer adapter for signing certificates.
        storage : TraceStoragePort, optional
            Storage adapter for persisting traces/certificates.
        """
        self._embedder = embedder
        self._kernel = kernel
        self._signer = signer
        self._storage = storage

    def generate_certificate(
        self,
        trace: Mapping[str, Any],
        *,
        psi_mode: str = "embedding",
        rank: int = 10,
        task_embedding: Optional[np.ndarray] = None,
        embedder_id: Optional[str] = None,
        kernel_mode: str = "prototype",
        precision_bits: int = 128,
        verify_with_kernel: bool = False,
    ) -> Dict[str, Any]:
        """Generate a spectral certificate from a trace.
        
        Parameters
        ----------
        trace : Mapping[str, Any]
            Trace payload containing embeddings or steps.
        psi_mode : str
            Embedding selection mode ("embedding" or "residual_stream").
        rank : int
            Rank for SVD truncation.
        task_embedding : np.ndarray, optional
            Task embedding used for semantic divergence.
        embedder_id : str, optional
            Embedder identifier for audit trails.
        kernel_mode : str
            Kernel mode (prototype/arb/mpfi).
        precision_bits : int
            Interval arithmetic precision.
        verify_with_kernel : bool
            Whether to use kernel verification.
            
        Returns
        -------
        Dict[str, Any]
            Certificate payload with optional kernel output.
        """
        # Use kernel adapter if verification requested and available
        kernel = self._kernel if verify_with_kernel else None
        
        # Generate certificate using domain logic
        certificate = compute_certificate_from_trace(
            trace,
            psi_mode=psi_mode,
            rank=rank,
            task_embedding=task_embedding,
            embedder_id=embedder_id,
            kernel=kernel,
            kernel_mode=kernel_mode,
            precision_bits=precision_bits,
        )
        
        return certificate

    def generate_and_sign(
        self,
        trace: Mapping[str, Any],
        *,
        psi_mode: str = "embedding",
        rank: int = 10,
        verify_with_kernel: bool = False,
    ) -> Dict[str, Any]:
        """Generate and sign a certificate.
        
        Parameters
        ----------
        trace : Mapping[str, Any]
            Trace payload.
        psi_mode : str
            Embedding selection mode.
        rank : int
            Rank for SVD truncation.
        verify_with_kernel : bool
            Whether to use kernel verification.
            
        Returns
        -------
        Dict[str, Any]
            Certificate with signature metadata.
            
        Raises
        ------
        RuntimeError
            If signer is not configured.
        """
        if self._signer is None:
            raise RuntimeError("Signer not configured for this service")
        
        # Generate certificate
        certificate = self.generate_certificate(
            trace,
            psi_mode=psi_mode,
            rank=rank,
            verify_with_kernel=verify_with_kernel,
        )
        
        # Sign the certificate
        import json
        certificate_bytes = json.dumps(certificate, sort_keys=True).encode("utf-8")
        signature = self._signer.sign_bytes(certificate_bytes)
        
        # Add signature metadata
        certificate["signature"] = {
            "signer_id": self._signer.signer_id,
            "signature_bytes": signature.hex(),
        }
        
        return certificate

    def generate_and_persist(
        self,
        trace: Mapping[str, Any],
        *,
        psi_mode: str = "embedding",
        rank: int = 10,
        verify_with_kernel: bool = False,
    ) -> Dict[str, Any]:
        """Generate a certificate and persist it to storage.
        
        Parameters
        ----------
        trace : Mapping[str, Any]
            Trace payload.
        psi_mode : str
            Embedding selection mode.
        rank : int
            Rank for SVD truncation.
        verify_with_kernel : bool
            Whether to use kernel verification.
            
        Returns
        -------
        Dict[str, Any]
            Certificate with storage metadata.
            
        Raises
        ------
        RuntimeError
            If storage is not configured.
        """
        if self._storage is None:
            raise RuntimeError("Storage not configured for this service")
        
        # Generate certificate
        certificate = self.generate_certificate(
            trace,
            psi_mode=psi_mode,
            rank=rank,
            verify_with_kernel=verify_with_kernel,
        )
        
        # Persist to storage
        # Note: This saves the original trace, not the certificate
        # In a real system, you might want a separate certificate storage
        storage_path = self._storage.save_trace(
            metadata=trace.get("metadata", {}),
            steps=trace.get("steps", []),
            outcome=certificate,
        )
        
        certificate["storage_path"] = str(storage_path)
        
        return certificate


__all__ = ["CertificateService"]
