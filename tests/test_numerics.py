"""Tests for numerical configuration module."""
from __future__ import annotations

import pytest

from certificates.numerics import (
    NumericalConfig,
    get_config,
    set_config,
    reset_config,
    get_high_precision_config,
    get_fast_config,
    get_robust_config,
)

# Mark all tests in this module as unit tests (Tier 1)
pytestmark = pytest.mark.unit


def test_default_config() -> None:
    """Test that default configuration has expected values."""
    config = NumericalConfig()
    assert config.eps == 1e-12
    assert config.ridge_regularization == 1e-6
    assert config.cond_thresh == 1e4
    assert config.gap_thresh == 1e-12
    assert config.relative_gap_thresh == 1e-6


def test_config_validation() -> None:
    """Test that invalid configuration values are rejected."""
    # Negative eps
    with pytest.raises(ValueError, match="eps must be positive"):
        NumericalConfig(eps=-1e-12)
    
    # Negative regularization
    with pytest.raises(ValueError, match="ridge_regularization must be non-negative"):
        NumericalConfig(ridge_regularization=-1e-6)
    
    # Invalid condition threshold
    with pytest.raises(ValueError, match="cond_thresh must be > 1"):
        NumericalConfig(cond_thresh=0.5)
    
    # Invalid explained variance
    with pytest.raises(ValueError, match="explained_variance_threshold"):
        NumericalConfig(explained_variance_threshold=1.5)
    
    # Invalid OOS folds
    with pytest.raises(ValueError, match="oos_min_folds"):
        NumericalConfig(oos_min_folds=0)
    
    with pytest.raises(ValueError, match="oos_max_folds"):
        NumericalConfig(oos_min_folds=5, oos_max_folds=3)


def test_config_to_dict() -> None:
    """Test configuration serialization to dictionary."""
    config = NumericalConfig()
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert config_dict["eps"] == 1e-12
    assert config_dict["cond_thresh"] == 1e4
    assert "ridge_regularization" in config_dict


def test_config_from_dict() -> None:
    """Test configuration deserialization from dictionary."""
    config_dict = {
        "eps": 1e-10,
        "ridge_regularization": 1e-5,
        "cond_thresh": 1e5,
        "gap_thresh": 1e-10,
        "relative_gap_thresh": 1e-5,
    }
    config = NumericalConfig.from_dict(config_dict)
    
    assert config.eps == 1e-10
    assert config.ridge_regularization == 1e-5
    assert config.cond_thresh == 1e5


def test_get_set_config() -> None:
    """Test global configuration get/set."""
    # Save original config
    original = get_config()
    
    try:
        # Set custom config
        custom = NumericalConfig(eps=1e-10)
        set_config(custom)
        
        # Verify it's active
        assert get_config().eps == 1e-10
        
        # Reset to default
        reset_config()
        assert get_config().eps == 1e-12
    finally:
        # Restore original config
        set_config(original)


def test_high_precision_config() -> None:
    """Test high precision configuration preset."""
    config = get_high_precision_config()
    
    # Should have tighter tolerances
    assert config.eps <= 1e-12
    assert config.cond_thresh <= 1e4
    assert config.ridge_regularization <= 1e-6


def test_fast_config() -> None:
    """Test fast configuration preset."""
    config = get_fast_config()
    
    # Should have relaxed tolerances
    assert config.cond_thresh >= 1e4
    assert config.randomized_svd_oversamples < 10
    assert config.randomized_svd_n_iter < 4


def test_robust_config() -> None:
    """Test robust configuration preset."""
    config = get_robust_config()
    
    # Should have higher regularization
    assert config.ridge_regularization >= 1e-6
    assert config.cond_thresh >= 1e4


def test_config_immutability_after_to_dict() -> None:
    """Test that modifying dict doesn't affect original config."""
    config = NumericalConfig()
    config_dict = config.to_dict()
    
    # Modify dict
    config_dict["eps"] = 999.0
    
    # Original config should be unchanged
    assert config.eps == 1e-12


def test_config_round_trip() -> None:
    """Test that config can be serialized and deserialized without loss."""
    original = NumericalConfig(
        eps=1e-10,
        ridge_regularization=1e-5,
        cond_thresh=1e5,
    )
    
    # Round trip through dict
    config_dict = original.to_dict()
    restored = NumericalConfig.from_dict(config_dict)
    
    # Should be equivalent
    assert restored.eps == original.eps
    assert restored.ridge_regularization == original.ridge_regularization
    assert restored.cond_thresh == original.cond_thresh
