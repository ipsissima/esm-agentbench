"""Test Guards and Tiered Test Infrastructure.

This module provides pytest markers and fixtures for tiered testing:

Tier 1: Unit tests (fast, no native code, no model loading)
    - Run in PR CI
    - Test Python logic, shims, fallback algorithms
    - Guards: ESM_ALLOW_KERNEL_LOAD=0, ESM_ALLOW_MODEL_LOAD=0

Tier 2: Integration tests (isolated container, kernel allowed)
    - Run in merge/integration CI
    - Test kernel operations, model loading with tiny models
    - Guards: ESM_ALLOW_KERNEL_LOAD=1, Docker container

Tier 3: Heavy tests (GPU, full models, attestation)
    - Run in nightly CI
    - Test full Phi-3, GPU paths, numeric verification
    - Guards: Requires CUDA, full model access

Usage in tests:
    @pytest.mark.unit
    def test_something_fast():
        ...

    @pytest.mark.integration
    def test_kernel_operations():
        ...

    @pytest.mark.heavy
    @pytest.mark.gpu
    def test_full_model_generation():
        ...
"""
from __future__ import annotations

import os
import sys
from functools import wraps
from typing import Any, Callable, Optional

import pytest

# Environment variables for test guards
ENV_ALLOW_KERNEL_LOAD = "ESM_ALLOW_KERNEL_LOAD"
ENV_ALLOW_MODEL_LOAD = "ESM_ALLOW_MODEL_LOAD"
ENV_STRICT_MODE = "ESM_STRICT"
ENV_KERNEL_SERVICE = "ESM_KERNEL_SERVICE"


def is_kernel_allowed() -> bool:
    """Check if kernel loading is allowed."""
    return os.environ.get(ENV_ALLOW_KERNEL_LOAD, "1") == "1"


def is_model_allowed() -> bool:
    """Check if model loading is allowed."""
    return os.environ.get(ENV_ALLOW_MODEL_LOAD, "1") == "1"


def is_strict_mode() -> bool:
    """Check if strict mode is enabled."""
    return os.environ.get(ENV_STRICT_MODE, "0") == "1"


def is_kernel_service_mode() -> bool:
    """Check if kernel-as-service mode is enabled."""
    return os.environ.get(ENV_KERNEL_SERVICE, "0") == "1"


def has_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def has_transformers() -> bool:
    """Check if transformers is installed."""
    try:
        import transformers
        return True
    except ImportError:
        return False


def has_pynacl() -> bool:
    """Check if PyNaCl is installed."""
    try:
        import nacl
        return True
    except ImportError:
        return False


# Pytest markers
unit = pytest.mark.unit
integration = pytest.mark.integration
heavy = pytest.mark.heavy
gpu = pytest.mark.gpu
kernel = pytest.mark.kernel
model = pytest.mark.model
signing = pytest.mark.signing


# Skip conditions
skip_if_no_kernel = pytest.mark.skipif(
    not is_kernel_allowed(),
    reason="Kernel loading disabled (ESM_ALLOW_KERNEL_LOAD=0)"
)

skip_if_no_model = pytest.mark.skipif(
    not is_model_allowed(),
    reason="Model loading disabled (ESM_ALLOW_MODEL_LOAD=0)"
)

skip_if_no_cuda = pytest.mark.skipif(
    not has_cuda(),
    reason="CUDA not available"
)

skip_if_no_transformers = pytest.mark.skipif(
    not has_transformers(),
    reason="transformers not installed"
)

skip_if_no_pynacl = pytest.mark.skipif(
    not has_pynacl(),
    reason="PyNaCl not installed"
)


def requires_kernel(func: Callable) -> Callable:
    """Decorator to skip test if kernel loading is not allowed."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not is_kernel_allowed():
            pytest.skip("Kernel loading disabled")
        return func(*args, **kwargs)
    return wrapper


def requires_model(func: Callable) -> Callable:
    """Decorator to skip test if model loading is not allowed."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not is_model_allowed():
            pytest.skip("Model loading disabled")
        return func(*args, **kwargs)
    return wrapper


def requires_cuda(func: Callable) -> Callable:
    """Decorator to skip test if CUDA is not available."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not has_cuda():
            pytest.skip("CUDA not available")
        return func(*args, **kwargs)
    return wrapper


def requires_strict(func: Callable) -> Callable:
    """Decorator to skip test if strict mode is not enabled."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not is_strict_mode():
            pytest.skip("Strict mode not enabled (ESM_STRICT=0)")
        return func(*args, **kwargs)
    return wrapper


# Test tier fixtures
@pytest.fixture
def disable_kernel_load(monkeypatch: Any) -> None:
    """Fixture to disable kernel loading for a test."""
    monkeypatch.setenv(ENV_ALLOW_KERNEL_LOAD, "0")


@pytest.fixture
def disable_model_load(monkeypatch: Any) -> None:
    """Fixture to disable model loading for a test."""
    monkeypatch.setenv(ENV_ALLOW_MODEL_LOAD, "0")


@pytest.fixture
def enable_strict_mode(monkeypatch: Any) -> None:
    """Fixture to enable strict mode for a test."""
    monkeypatch.setenv(ENV_STRICT_MODE, "1")


@pytest.fixture
def enable_kernel_service(monkeypatch: Any) -> None:
    """Fixture to enable kernel-as-service mode for a test."""
    monkeypatch.setenv(ENV_KERNEL_SERVICE, "1")


@pytest.fixture
def unit_test_env(monkeypatch: Any) -> None:
    """Fixture for unit test environment (no native code)."""
    monkeypatch.setenv(ENV_ALLOW_KERNEL_LOAD, "0")
    monkeypatch.setenv(ENV_ALLOW_MODEL_LOAD, "0")


@pytest.fixture
def integration_test_env(monkeypatch: Any) -> None:
    """Fixture for integration test environment."""
    monkeypatch.setenv(ENV_ALLOW_KERNEL_LOAD, "1")
    monkeypatch.setenv(ENV_ALLOW_MODEL_LOAD, "1")
    monkeypatch.setenv(ENV_STRICT_MODE, "1")


class TestTierManager:
    """Manager for test tier configuration.

    Provides methods to check current tier and configure environment.
    """

    TIER_UNIT = 1
    TIER_INTEGRATION = 2
    TIER_HEAVY = 3

    @classmethod
    def current_tier(cls) -> int:
        """Get the current test tier based on environment."""
        if not is_kernel_allowed() and not is_model_allowed():
            return cls.TIER_UNIT
        elif has_cuda():
            return cls.TIER_HEAVY
        else:
            return cls.TIER_INTEGRATION

    @classmethod
    def configure_for_tier(cls, tier: int) -> None:
        """Configure environment for a specific tier."""
        if tier == cls.TIER_UNIT:
            os.environ[ENV_ALLOW_KERNEL_LOAD] = "0"
            os.environ[ENV_ALLOW_MODEL_LOAD] = "0"
            os.environ[ENV_STRICT_MODE] = "0"
        elif tier == cls.TIER_INTEGRATION:
            os.environ[ENV_ALLOW_KERNEL_LOAD] = "1"
            os.environ[ENV_ALLOW_MODEL_LOAD] = "1"
            os.environ[ENV_STRICT_MODE] = "1"
        elif tier == cls.TIER_HEAVY:
            os.environ[ENV_ALLOW_KERNEL_LOAD] = "1"
            os.environ[ENV_ALLOW_MODEL_LOAD] = "1"
            os.environ[ENV_STRICT_MODE] = "1"

    @classmethod
    def tier_name(cls, tier: int) -> str:
        """Get human-readable tier name."""
        names = {
            cls.TIER_UNIT: "unit",
            cls.TIER_INTEGRATION: "integration",
            cls.TIER_HEAVY: "heavy",
        }
        return names.get(tier, "unknown")


# Register markers for pytest
def pytest_configure(config: Any) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test (tier 1, fast, no native code)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (tier 2, container)"
    )
    config.addinivalue_line(
        "markers", "heavy: mark test as heavy test (tier 3, GPU, full models)"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "kernel: mark test as requiring kernel loading"
    )
    config.addinivalue_line(
        "markers", "model: mark test as requiring model loading"
    )
    config.addinivalue_line(
        "markers", "signing: mark test as requiring signing capabilities"
    )


def pytest_collection_modifyitems(config: Any, items: Any) -> None:
    """Automatically skip tests based on current tier."""
    tier = TestTierManager.current_tier()

    for item in items:
        # Skip heavy tests in non-heavy tiers
        if "heavy" in item.keywords and tier < TestTierManager.TIER_HEAVY:
            item.add_marker(
                pytest.mark.skip(reason=f"Heavy test skipped in {TestTierManager.tier_name(tier)} tier")
            )

        # Skip integration tests in unit tier
        if "integration" in item.keywords and tier < TestTierManager.TIER_INTEGRATION:
            item.add_marker(
                pytest.mark.skip(reason=f"Integration test skipped in {TestTierManager.tier_name(tier)} tier")
            )

        # Skip kernel tests if kernel not allowed
        if "kernel" in item.keywords and not is_kernel_allowed():
            item.add_marker(
                pytest.mark.skip(reason="Kernel loading disabled")
            )

        # Skip model tests if model not allowed
        if "model" in item.keywords and not is_model_allowed():
            item.add_marker(
                pytest.mark.skip(reason="Model loading disabled")
            )

        # Skip GPU tests if no CUDA
        if "gpu" in item.keywords and not has_cuda():
            item.add_marker(
                pytest.mark.skip(reason="CUDA not available")
            )

        # Skip signing tests if no PyNaCl
        if "signing" in item.keywords and not has_pynacl():
            item.add_marker(
                pytest.mark.skip(reason="PyNaCl not installed")
            )


__all__ = [
    # Environment checks
    "is_kernel_allowed",
    "is_model_allowed",
    "is_strict_mode",
    "is_kernel_service_mode",
    "has_cuda",
    "has_transformers",
    "has_pynacl",
    # Markers
    "unit",
    "integration",
    "heavy",
    "gpu",
    "kernel",
    "model",
    "signing",
    # Skip conditions
    "skip_if_no_kernel",
    "skip_if_no_model",
    "skip_if_no_cuda",
    "skip_if_no_transformers",
    "skip_if_no_pynacl",
    # Decorators
    "requires_kernel",
    "requires_model",
    "requires_cuda",
    "requires_strict",
    # Manager
    "TestTierManager",
]
