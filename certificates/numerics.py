"""Numerical configuration and constants for certificate computation.

This module centralizes all numerical constants used throughout the certificate
computation pipeline, including thresholds, tolerances, and epsilon values.

The constants here interact with the formally verified Coq constants in the
kernel. Specifically:

1. **Condition number threshold (cond_thresh)**: Limits numerical instability
   in matrix operations. Maps to Coq's condition number bounds.

2. **Spectral gap thresholds (gap_thresh, relative_gap_thresh)**: Ensure 
   singular values are sufficiently separated for stable subspace computations.
   Related to Wedin's theorem stability constants.

3. **Regularization (ridge_regularization)**: Added to Gram matrices to prevent
   singularity. Must be small enough to not bias results but large enough for
   numerical stability.

4. **Epsilon floors (eps, svd_eps)**: Prevent division by zero and ensure
   well-defined logarithms. Should be orders of magnitude smaller than expected
   signal magnitudes.

5. **OOS fold parameters**: Control out-of-sample validation for temporal
   operators, ensuring meaningful cross-validation without excessive holdout.

Changes to these values affect reproducibility and should be documented in
experiments.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class NumericalConfig:
    """Numerical configuration for certificate computation.
    
    Attributes
    ----------
    eps : float
        General epsilon for numerical stability (division by zero, etc.).
        Default: 1e-12
    
    svd_eps : float
        Epsilon for SVD-related computations (singular value floors, etc.).
        Default: 1e-12
    
    ridge_regularization : float
        Ridge regularization parameter for temporal operator fitting.
        Default: 1e-6
    
    cond_thresh : float
        Maximum allowed condition number for witness matrices.
        Default: 1e4
    
    gap_thresh : float
        Minimum absolute spectral gap (sigma_1 - sigma_2).
        Default: 1e-12
    
    relative_gap_thresh : float
        Minimum relative spectral gap (gap / sigma_1).
        Prevents false positives when all singular values are tiny.
        Default: 1e-6
    
    oos_min_folds : int
        Minimum number of out-of-sample folds for temporal validation.
        Default: 1
    
    oos_max_folds : int
        Maximum number of out-of-sample folds.
        Default: 3
    
    oos_max_fraction : float
        Maximum fraction of data to use for OOS validation.
        Default: 0.25 (25%)
    
    explained_variance_threshold : float
        Minimum cumulative explained variance for rank selection.
        Default: 0.95 (95%)
    
    min_timesteps_for_oos : int
        Minimum number of timesteps required for OOS validation.
        Default: 4
    
    randomized_svd_oversamples : int
        Number of oversamples for randomized SVD (if used).
        Default: 10
    
    randomized_svd_n_iter : int
        Number of power iterations for randomized SVD.
        Default: 4
    """
    
    # General numerical stability
    eps: float = 1e-12
    svd_eps: float = 1e-12
    
    # Regularization
    ridge_regularization: float = 1e-6
    
    # Witness validation thresholds
    cond_thresh: float = 1e4
    gap_thresh: float = 1e-12
    relative_gap_thresh: float = 1e-6
    
    # Out-of-sample validation
    oos_min_folds: int = 1
    oos_max_folds: int = 3
    oos_max_fraction: float = 0.25
    min_timesteps_for_oos: int = 4
    
    # Rank selection
    explained_variance_threshold: float = 0.95
    
    # Randomized SVD parameters
    randomized_svd_oversamples: int = 10
    randomized_svd_n_iter: int = 4
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.ridge_regularization < 0:
            raise ValueError("ridge_regularization must be non-negative")
        if self.cond_thresh <= 1:
            raise ValueError("cond_thresh must be > 1")
        if self.gap_thresh < 0:
            raise ValueError("gap_thresh must be non-negative")
        if self.relative_gap_thresh < 0:
            raise ValueError("relative_gap_thresh must be non-negative")
        if not 0 < self.explained_variance_threshold <= 1:
            raise ValueError("explained_variance_threshold must be in (0, 1]")
        if self.oos_min_folds < 1:
            raise ValueError("oos_min_folds must be >= 1")
        if self.oos_max_folds < self.oos_min_folds:
            raise ValueError("oos_max_folds must be >= oos_min_folds")
        if not 0 < self.oos_max_fraction < 1:
            raise ValueError("oos_max_fraction must be in (0, 1)")
    
    def to_dict(self) -> dict:
        """Export configuration as dictionary for logging/serialization."""
        return {
            "eps": self.eps,
            "svd_eps": self.svd_eps,
            "ridge_regularization": self.ridge_regularization,
            "cond_thresh": self.cond_thresh,
            "gap_thresh": self.gap_thresh,
            "relative_gap_thresh": self.relative_gap_thresh,
            "oos_min_folds": self.oos_min_folds,
            "oos_max_folds": self.oos_max_folds,
            "oos_max_fraction": self.oos_max_fraction,
            "min_timesteps_for_oos": self.min_timesteps_for_oos,
            "explained_variance_threshold": self.explained_variance_threshold,
            "randomized_svd_oversamples": self.randomized_svd_oversamples,
            "randomized_svd_n_iter": self.randomized_svd_n_iter,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "NumericalConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)


# Default global configuration
DEFAULT_CONFIG = NumericalConfig()


def get_config() -> NumericalConfig:
    """Get the current numerical configuration.
    
    Returns
    -------
    NumericalConfig
        The active numerical configuration.
    """
    return DEFAULT_CONFIG


def set_config(config: NumericalConfig) -> None:
    """Set the global numerical configuration.
    
    Parameters
    ----------
    config : NumericalConfig
        The new configuration to use globally.
        
    Warning
    -------
    Changing the configuration affects reproducibility. Document changes
    in experiment logs.
    """
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config


def reset_config() -> None:
    """Reset configuration to default values."""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = NumericalConfig()


# Convenience functions for common configuration scenarios
def get_high_precision_config() -> NumericalConfig:
    """Configuration for high-precision computation.
    
    Uses tighter thresholds and more conservative parameters.
    Suitable for critical applications where numerical errors must be minimized.
    """
    return NumericalConfig(
        eps=1e-15,
        svd_eps=1e-15,
        ridge_regularization=1e-8,
        cond_thresh=1e3,
        gap_thresh=1e-10,
        relative_gap_thresh=1e-5,
    )


def get_fast_config() -> NumericalConfig:
    """Configuration for fast computation.
    
    Uses relaxed thresholds and fewer iterations.
    Suitable for exploratory analysis or large-scale experiments.
    """
    return NumericalConfig(
        eps=1e-10,
        svd_eps=1e-10,
        ridge_regularization=1e-5,
        cond_thresh=1e5,
        gap_thresh=1e-14,
        relative_gap_thresh=1e-7,
        randomized_svd_oversamples=5,
        randomized_svd_n_iter=2,
    )


def get_robust_config() -> NumericalConfig:
    """Configuration for robust computation with challenging data.
    
    Uses higher regularization and more lenient thresholds.
    Suitable for noisy data or ill-conditioned problems.
    """
    return NumericalConfig(
        eps=1e-10,
        ridge_regularization=1e-4,
        cond_thresh=1e6,
        gap_thresh=1e-14,
        relative_gap_thresh=1e-8,
    )


__all__ = [
    "NumericalConfig",
    "DEFAULT_CONFIG",
    "get_config",
    "set_config",
    "reset_config",
    "get_high_precision_config",
    "get_fast_config",
    "get_robust_config",
]
