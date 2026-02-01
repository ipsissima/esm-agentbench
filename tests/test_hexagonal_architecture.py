"""Tests for adapter factory and service layer."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest


class TestAdapterFactory:
    """Test suite for AdapterFactory."""

    def test_create_embedder_default(self):
        """Factory should create sentence-transformers embedder."""
        from services.adapter_factory import AdapterFactory
        from ports.embedder import EmbedderPort

        embedder = AdapterFactory.create_embedder()
        assert isinstance(embedder, EmbedderPort)

    def test_create_embedder_custom_model(self):
        """Factory should create embedder with custom model name."""
        from services.adapter_factory import AdapterFactory
        from adapters import SentenceTransformersEmbedder

        embedder = AdapterFactory.create_embedder("custom-model")
        assert isinstance(embedder, SentenceTransformersEmbedder)
        assert embedder._model_name == "custom-model"

    def test_create_embedder_invalid_type(self):
        """Factory should raise ValueError for unsupported adapter type."""
        from services.adapter_factory import AdapterFactory

        with pytest.raises(ValueError, match="Unsupported embedder adapter type"):
            AdapterFactory.create_embedder(adapter_type="invalid")

    def test_create_inference_adapter(self):
        """Factory should create HuggingFace inference adapter."""
        from services.adapter_factory import AdapterFactory
        from ports.inference import InferencePort

        adapter = AdapterFactory.create_inference_adapter("test-model")
        assert isinstance(adapter, InferencePort)

    def test_create_inference_adapter_with_config(self):
        """Factory should create inference adapter with config path."""
        from services.adapter_factory import AdapterFactory
        from adapters import HuggingFaceInferenceAdapter

        config_path = Path("/tmp/models.yaml")
        adapter = AdapterFactory.create_inference_adapter(
            "test-model",
            config_path=config_path,
        )
        assert isinstance(adapter, HuggingFaceInferenceAdapter)
        assert adapter._config_path == config_path

    def test_create_inference_adapter_invalid_type(self):
        """Factory should raise ValueError for unsupported adapter type."""
        from services.adapter_factory import AdapterFactory

        with pytest.raises(ValueError, match="Unsupported inference adapter type"):
            AdapterFactory.create_inference_adapter("test", adapter_type="invalid")

    def test_create_kernel_adapter(self):
        """Factory should create kernel adapter."""
        from services.adapter_factory import AdapterFactory
        from ports.kernel import KernelClientPort

        adapter = AdapterFactory.create_kernel_adapter()
        assert isinstance(adapter, KernelClientPort)

    def test_create_kernel_adapter_invalid_type(self):
        """Factory should raise ValueError for unsupported adapter type."""
        from services.adapter_factory import AdapterFactory

        with pytest.raises(ValueError, match="Unsupported kernel adapter type"):
            AdapterFactory.create_kernel_adapter(adapter_type="invalid")

    def test_create_signer(self):
        """Factory should create GPG signer adapter."""
        from services.adapter_factory import AdapterFactory
        from ports.signer import SignerPort

        signer = AdapterFactory.create_signer("test-signer", "test-key")
        assert isinstance(signer, SignerPort)
        assert signer.signer_id == "test-signer"

    def test_create_signer_invalid_type(self):
        """Factory should raise ValueError for unsupported adapter type."""
        from services.adapter_factory import AdapterFactory

        with pytest.raises(ValueError, match="Unsupported signer adapter type"):
            AdapterFactory.create_signer("test", "key", adapter_type="invalid")

    def test_create_storage(self):
        """Factory should create filesystem storage adapter."""
        from services.adapter_factory import AdapterFactory
        from ports.storage import TraceStoragePort

        storage = AdapterFactory.create_storage(Path("/tmp/traces"))
        assert isinstance(storage, TraceStoragePort)

    def test_create_storage_invalid_type(self):
        """Factory should raise ValueError for unsupported adapter type."""
        from services.adapter_factory import AdapterFactory

        with pytest.raises(ValueError, match="Unsupported storage adapter type"):
            AdapterFactory.create_storage(Path("/tmp"), adapter_type="invalid")

    def test_convenience_functions(self):
        """Convenience functions should work as shortcuts."""
        from services.adapter_factory import (
            create_embedder,
            create_inference_adapter,
            create_kernel_adapter,
            create_signer,
            create_storage,
        )
        from ports.embedder import EmbedderPort
        from ports.inference import InferencePort
        from ports.kernel import KernelPort
        from ports.signer import SignerPort
        from ports.storage import TraceStoragePort

        assert isinstance(create_embedder(), EmbedderPort)
        assert isinstance(create_inference_adapter("test"), InferencePort)
        assert isinstance(create_kernel_adapter(), KernelPort)
        assert isinstance(create_signer("id", "key"), SignerPort)
        assert isinstance(create_storage(Path("/tmp")), TraceStoragePort)


class TestCertificateService:
    """Test suite for CertificateService."""

    def test_init_without_adapters(self):
        """Service should initialize without adapters."""
        from services.certificate_service import CertificateService

        service = CertificateService()
        assert service._embedder is None
        assert service._kernel is None
        assert service._signer is None
        assert service._storage is None

    def test_init_with_adapters(self):
        """Service should initialize with adapters."""
        from services.certificate_service import CertificateService

        mock_embedder = mock.Mock()
        mock_kernel = mock.Mock()
        mock_signer = mock.Mock()
        mock_storage = mock.Mock()

        service = CertificateService(
            embedder=mock_embedder,
            kernel=mock_kernel,
            signer=mock_signer,
            storage=mock_storage,
        )

        assert service._embedder is mock_embedder
        assert service._kernel is mock_kernel
        assert service._signer is mock_signer
        assert service._storage is mock_storage

    def test_generate_certificate_basic(self):
        """Service should generate certificate from trace."""
        from services.certificate_service import CertificateService

        # Create a simple trace with embeddings
        trace = {
            "embeddings": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        }

        service = CertificateService()
        certificate = service.generate_certificate(trace, rank=2)

        assert "residual" in certificate
        assert "theoretical_bound" in certificate
        assert isinstance(certificate["residual"], float)
        assert isinstance(certificate["theoretical_bound"], float)

    def test_generate_certificate_with_kernel(self):
        """Service should use kernel when verify_with_kernel=True."""
        from services.certificate_service import CertificateService

        mock_kernel = mock.Mock()
        mock_kernel.run_kernel_and_verify.return_value = {"status": "verified"}

        trace = {
            "embeddings": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        }

        service = CertificateService(kernel=mock_kernel)
        certificate = service.generate_certificate(
            trace,
            rank=2,
            verify_with_kernel=True,
        )

        # Kernel should be called
        assert mock_kernel.run_kernel_and_verify.called
        assert "kernel_output" in certificate

    def test_generate_and_sign_without_signer(self):
        """Service should raise error if signer not configured."""
        from services.certificate_service import CertificateService

        trace = {
            "embeddings": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        }

        service = CertificateService()
        with pytest.raises(RuntimeError, match="Signer not configured"):
            service.generate_and_sign(trace)

    def test_generate_and_sign_with_signer(self):
        """Service should sign certificate when signer configured."""
        from services.certificate_service import CertificateService

        mock_signer = mock.Mock()
        mock_signer.signer_id = "test-signer"
        mock_signer.sign_bytes.return_value = b"\x00\x01\x02\x03"

        trace = {
            "embeddings": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        }

        service = CertificateService(signer=mock_signer)
        certificate = service.generate_and_sign(trace)

        assert "signature" in certificate
        assert certificate["signature"]["signer_id"] == "test-signer"
        assert mock_signer.sign_bytes.called

    def test_generate_and_persist_without_storage(self):
        """Service should raise error if storage not configured."""
        from services.certificate_service import CertificateService

        trace = {
            "embeddings": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        }

        service = CertificateService()
        with pytest.raises(RuntimeError, match="Storage not configured"):
            service.generate_and_persist(trace)

    def test_generate_and_persist_with_storage(self):
        """Service should persist certificate when storage configured."""
        from services.certificate_service import CertificateService

        mock_storage = mock.Mock()
        mock_storage.save_trace.return_value = Path("/tmp/trace.json")

        trace = {
            "metadata": {"trace_id": "test"},
            "steps": [],
            "embeddings": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        }

        service = CertificateService(storage=mock_storage)
        certificate = service.generate_and_persist(trace)

        assert "storage_path" in certificate
        assert certificate["storage_path"] == "/tmp/trace.json"
        assert mock_storage.save_trace.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
