"""Service layer for hexagonal architecture.

This module provides service facades and dependency injection utilities
for managing port implementations (adapters) in a clean, testable way.
"""

from .adapter_factory import (
    AdapterFactory,
    create_embedder,
    create_inference_adapter,
    create_kernel_adapter,
    create_signer,
    create_storage,
)
from .certificate_service import CertificateService

__all__ = [
    "AdapterFactory",
    "CertificateService",
    "create_embedder",
    "create_inference_adapter",
    "create_kernel_adapter",
    "create_signer",
    "create_storage",
]
