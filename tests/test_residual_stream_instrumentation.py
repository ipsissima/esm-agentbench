"""Tests for residual stream instrumentation in inference.py.

These tests verify that the TransformersBackend can capture
internal model states (residual stream) during generation.
"""
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mark all tests as unit tests
pytestmark = pytest.mark.unit


class MockTensor:
    """Mock PyTorch tensor for testing."""

    def __init__(self, data):
        self._data = np.array(data)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._data

    def __getitem__(self, idx):
        return MockTensor(self._data[idx])

    @property
    def shape(self):
        return self._data.shape


class MockModule:
    """Mock PyTorch module for testing hooks."""

    def __init__(self, name):
        self.name = name
        self._hooks = []

    def register_forward_hook(self, hook):
        handle = MagicMock()
        handle.remove = MagicMock()
        self._hooks.append((hook, handle))
        return handle

    def simulate_forward(self, output):
        """Simulate a forward pass with the given output."""
        for hook, _ in self._hooks:
            hook(self, None, output)


class MockModel:
    """Mock transformer model for testing."""

    def __init__(self):
        self.device = "cpu"
        self.config = MagicMock()
        self.layers = {
            "model.layers.0": MockModule("layer0"),
            "model.layers.1": MockModule("layer1"),
            "model.layers.2": MockModule("layer2"),
        }

    def named_modules(self):
        for name, module in self.layers.items():
            yield name, module

    def generate(self, **kwargs):
        # Simulate forward pass through layers
        batch_size = 1
        seq_len = 10
        hidden_dim = 64

        for name, layer in self.layers.items():
            # Simulate output: [batch, seq_len, hidden_dim]
            output = MockTensor(np.random.randn(batch_size, seq_len, hidden_dim))
            layer.simulate_forward(output)

        # Return mock output
        return [np.array([1, 2, 3, 4, 5])]


class TestResidualStreamInstrumentation:
    """Tests for residual stream capture."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock TransformersBackend."""
        from tools.real_agents_hf.inference import ModelConfig, TransformersBackend

        config = ModelConfig(
            name="test-model",
            hf_id="test/model",
            backend="transformers",
            dtype="float32",
            max_tokens=100,
            temperature=0.7,
        )

        backend = TransformersBackend(config)
        backend.model = MockModel()
        backend.tokenizer = MagicMock()
        backend.tokenizer.pad_token = "<pad>"
        backend.tokenizer.eos_token = "</s>"
        backend.tokenizer.pad_token_id = 0
        backend.tokenizer.eos_token_id = 1
        backend.tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": np.array([[1, 1, 1]]),
        }
        backend.tokenizer.decode = MagicMock(return_value="test output")

        return backend

    def test_register_hooks(self, mock_backend):
        """Test that hooks can be registered on model layers."""
        mock_backend._register_residual_hooks()

        # Check hooks were registered
        assert len(mock_backend._residual_hooks) > 0

        # Clean up
        mock_backend._remove_residual_hooks()
        assert len(mock_backend._residual_hooks) == 0

    def test_capture_residuals(self, mock_backend):
        """Test that residual stream is captured during forward."""
        mock_backend._register_residual_hooks()

        # Simulate forward pass
        for name, layer in mock_backend.model.layers.items():
            output = MockTensor(np.random.randn(1, 10, 64))
            layer.simulate_forward(output)

        # Check residuals were captured
        residuals = mock_backend.get_residual_stream()
        assert len(residuals) > 0

        # Clean up
        mock_backend._remove_residual_hooks()
        mock_backend.clear_residual_stream()
        assert len(mock_backend.get_residual_stream()) == 0

    def test_residual_stream_shape(self, mock_backend):
        """Test that captured residuals have expected shape."""
        mock_backend._register_residual_hooks()

        # Simulate forward with known dimensions
        hidden_dim = 64
        for name, layer in mock_backend.model.layers.items():
            output = MockTensor(np.random.randn(1, 10, hidden_dim))
            layer.simulate_forward(output)

        residuals = mock_backend.get_residual_stream()

        # Each residual should be 1D (hidden_dim,) - last token representation
        for r in residuals:
            assert r.ndim == 1
            assert len(r) == hidden_dim

        mock_backend._remove_residual_hooks()

    def test_env_variable_control(self):
        """Test that ESM_CAPTURE_INTERNAL controls capture."""
        from tools.real_agents_hf.inference import ModelConfig, TransformersBackend

        config = ModelConfig(
            name="test-model",
            hf_id="test/model",
            backend="transformers",
            dtype="float32",
            max_tokens=100,
            temperature=0.7,
        )

        # Without env var
        with patch.dict(os.environ, {"ESM_CAPTURE_INTERNAL": "0"}):
            backend = TransformersBackend(config)
            assert backend._capture_residuals is False

        # With env var
        with patch.dict(os.environ, {"ESM_CAPTURE_INTERNAL": "1"}):
            backend = TransformersBackend(config)
            assert backend._capture_residuals is True


class TestResidualStreamWithMockGenerate:
    """Test residual stream capture during generation."""

    @pytest.fixture
    def backend_with_generate(self):
        """Create backend that can mock generate."""
        from tools.real_agents_hf.inference import ModelConfig, TransformersBackend

        config = ModelConfig(
            name="test-model",
            hf_id="test/model",
            backend="transformers",
            dtype="float32",
            max_tokens=100,
            temperature=0.0,  # Deterministic
        )

        backend = TransformersBackend(config)

        # Setup mocks
        mock_model = MockModel()
        backend.model = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1

        # Mock tokenizer call
        def tokenizer_call(*args, **kwargs):
            result = MagicMock()
            result.__getitem__ = lambda self, key: {
                "input_ids": MockTensor(np.array([[1, 2, 3]])),
                "attention_mask": MockTensor(np.array([[1, 1, 1]])),
            }.get(key, MockTensor(np.array([[1, 2, 3]])))
            result.items = lambda: [
                ("input_ids", MockTensor(np.array([[1, 2, 3]]))),
                ("attention_mask", MockTensor(np.array([[1, 1, 1]]))),
            ]
            return result

        mock_tokenizer.side_effect = tokenizer_call
        mock_tokenizer.decode = MagicMock(return_value="generated text")
        backend.tokenizer = mock_tokenizer

        return backend

    def test_generate_with_capture_flag(self, backend_with_generate):
        """Test generation with explicit capture_residuals flag."""
        # This test would require more complex mocking of torch
        # For now, just verify the method exists and handles the flag
        assert hasattr(backend_with_generate, "generate")

        # Check that capture_residuals parameter is accepted
        import inspect
        sig = inspect.signature(backend_with_generate.generate)
        assert "capture_residuals" in sig.parameters
