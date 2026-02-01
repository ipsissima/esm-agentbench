"""Adapter factory for creating port implementations.

This module provides factory functions for creating adapter instances,
enabling dependency injection and easier testing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ports.embedder import EmbedderPort
from ports.inference import InferencePort
from ports.kernel import KernelClientPort
from ports.signer import SignerPort
from ports.storage import TraceStoragePort


class AdapterFactory:
    """Factory for creating adapter instances.
    
    This class provides a centralized way to create adapters with
    consistent configuration and error handling.
    """

    @staticmethod
    def create_embedder(
        model_name: str = "all-MiniLM-L6-v2",
        *,
        adapter_type: str = "sentence-transformers",
    ) -> EmbedderPort:
        """Create an embedder adapter.
        
        Parameters
        ----------
        model_name : str
            Name of the embedding model to use.
        adapter_type : str
            Type of adapter ("sentence-transformers").
            
        Returns
        -------
        EmbedderPort
            An embedder adapter instance.
            
        Raises
        ------
        ValueError
            If adapter_type is not supported.
        """
        if adapter_type == "sentence-transformers":
            from adapters import SentenceTransformersEmbedder
            return SentenceTransformersEmbedder(model_name)
        else:
            raise ValueError(f"Unsupported embedder adapter type: {adapter_type}")

    @staticmethod
    def create_inference_adapter(
        model_name: str,
        *,
        config_path: Optional[Path] = None,
        adapter_type: str = "huggingface",
    ) -> InferencePort:
        """Create an inference adapter.
        
        Parameters
        ----------
        model_name : str
            Name of the model to use.
        config_path : Path, optional
            Path to model configuration file.
        adapter_type : str
            Type of adapter ("huggingface").
            
        Returns
        -------
        InferencePort
            An inference adapter instance.
            
        Raises
        ------
        ValueError
            If adapter_type is not supported.
        """
        if adapter_type == "huggingface":
            from adapters import HuggingFaceInferenceAdapter
            return HuggingFaceInferenceAdapter(model_name, config_path)
        else:
            raise ValueError(f"Unsupported inference adapter type: {adapter_type}")

    @staticmethod
    def create_kernel_adapter(
        *,
        adapter_type: str = "kernel-client",
    ) -> KernelClientPort:
        """Create a kernel client adapter.
        
        Parameters
        ----------
        adapter_type : str
            Type of adapter ("kernel-client").
            
        Returns
        -------
        KernelClientPort
            A kernel client adapter instance.
            
        Raises
        ------
        ValueError
            If adapter_type is not supported.
        """
        if adapter_type == "kernel-client":
            from adapters import KernelClientAdapter
            return KernelClientAdapter()
        else:
            raise ValueError(f"Unsupported kernel adapter type: {adapter_type}")

    @staticmethod
    def create_signer(
        signer_id: str,
        gpg_key: str,
        *,
        adapter_type: str = "gpg",
    ) -> SignerPort:
        """Create a signer adapter.
        
        Parameters
        ----------
        signer_id : str
            Identifier for the signer.
        gpg_key : str
            GPG key identifier.
        adapter_type : str
            Type of adapter ("gpg").
            
        Returns
        -------
        SignerPort
            A signer adapter instance.
            
        Raises
        ------
        ValueError
            If adapter_type is not supported.
        """
        if adapter_type == "gpg":
            from adapters import GpgSignerAdapter
            return GpgSignerAdapter(signer_id, gpg_key)
        else:
            raise ValueError(f"Unsupported signer adapter type: {adapter_type}")

    @staticmethod
    def create_storage(
        root_dir: Path,
        *,
        adapter_type: str = "filesystem",
    ) -> TraceStoragePort:
        """Create a storage adapter.
        
        Parameters
        ----------
        root_dir : Path
            Root directory for storage.
        adapter_type : str
            Type of adapter ("filesystem").
            
        Returns
        -------
        TraceStoragePort
            A storage adapter instance.
            
        Raises
        ------
        ValueError
            If adapter_type is not supported.
        """
        if adapter_type == "filesystem":
            from adapters import FilesystemTraceStorage
            return FilesystemTraceStorage(root_dir)
        else:
            raise ValueError(f"Unsupported storage adapter type: {adapter_type}")


__all__ = [
    "AdapterFactory",
]
