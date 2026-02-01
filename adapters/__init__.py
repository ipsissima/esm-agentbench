"""Infrastructure adapters for ports.

This module provides concrete implementations of the port interfaces
defined in the ports package, following the hexagonal architecture pattern.

Available Adapters:
- SentenceTransformersEmbedder: Embedder using sentence-transformers
- HuggingFaceInferenceAdapter: Inference using HuggingFace models
- KernelClientAdapter: Kernel execution adapter
- GpgSignerAdapter: GPG-based signing adapter
- FilesystemTraceStorage: Filesystem-based trace storage
"""

from .embedder_sentence_transformers import SentenceTransformersEmbedder
from .inference_huggingface import HuggingFaceInferenceAdapter
from .kernel_client import KernelClientAdapter
from .signer_gpg import GpgSignerAdapter
from .storage_fs import FilesystemTraceStorage

__all__ = [
    "SentenceTransformersEmbedder",
    "HuggingFaceInferenceAdapter",
    "KernelClientAdapter",
    "GpgSignerAdapter",
    "FilesystemTraceStorage",
]
