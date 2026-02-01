"""Configuration validation for ESM Assessor.

This module provides Pydantic-based configuration validation for runtime settings.
All environment variables and configuration options are validated at startup.
"""
from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class AssessorConfig(BaseModel):
    """Validated configuration for the ESM Assessor service."""

    # Server settings
    host: str = Field(default="127.0.0.1", description="Assessor server host")
    port: int = Field(default=8080, ge=1, le=65535, description="Assessor server port")

    # API settings
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key for LLM calls")

    # Embedding settings
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    force_tfidf: bool = Field(default=False, description="Force TF-IDF embeddings instead of transformers")
    skip_sentence_transformers: bool = Field(
        default=False,
        description="Skip loading sentence-transformers (use fallback)"
    )

    # Evaluation settings
    semantic_oracle_model: str = Field(
        default="gpt-4o-mini",
        description="Model for semantic oracle evaluation"
    )
    oracle_api_delay: float = Field(
        default=0.1,
        ge=0.0,
        description="Delay between API calls to avoid rate limits"
    )
    allow_deterministic_fallback: bool = Field(
        default=False,
        description="Allow fallback to deterministic agent when API fails"
    )

    # Debug settings
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    verbose: bool = Field(default=False, description="Enable verbose logging")

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def validate_api_key(cls, v: Optional[str]) -> Optional[str]:
        """Validate API key format if provided."""
        if v is not None and v.strip() == "":
            return None
        return v

    @classmethod
    def from_env(cls) -> "AssessorConfig":
        """Load configuration from environment variables.

        Environment variables:
            ASSESSOR_HOST: Server host (default: 127.0.0.1)
            ASSESSOR_PORT: Server port (default: 8080)
            OPENAI_API_KEY: OpenAI API key
            EMBEDDING_MODEL: Sentence transformer model name
            ESM_FORCE_TFIDF: Force TF-IDF embeddings (1/true/yes)
            SKIP_SENTENCE_TRANSFORMERS: Skip transformer loading (1/true/yes)
            SEMANTIC_ORACLE_MODEL: Model for semantic evaluation
            ORACLE_API_DELAY: Delay between API calls
            ALLOW_DETERMINISTIC_FALLBACK: Allow deterministic fallback (1/true/yes)
            DEBUG: Enable debug mode (1/true/yes)
            VERBOSE: Enable verbose logging (1/true/yes)

        Returns:
            AssessorConfig: Validated configuration object.
        """
        return cls(
            host=os.environ.get("ASSESSOR_HOST", "127.0.0.1"),
            port=int(os.environ.get("ASSESSOR_PORT", "8080")),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            force_tfidf=os.environ.get("ESM_FORCE_TFIDF", "").lower() in ("1", "true", "yes"),
            skip_sentence_transformers=os.environ.get(
                "SKIP_SENTENCE_TRANSFORMERS", ""
            ).lower() in ("1", "true", "yes"),
            semantic_oracle_model=os.environ.get("SEMANTIC_ORACLE_MODEL", "gpt-4o-mini"),
            oracle_api_delay=float(os.environ.get("ORACLE_API_DELAY", "0.1")),
            allow_deterministic_fallback=os.environ.get(
                "ALLOW_DETERMINISTIC_FALLBACK", ""
            ).lower() in ("1", "true", "yes"),
            debug_mode=os.environ.get("DEBUG", "").lower() in ("1", "true", "yes"),
            verbose=os.environ.get("VERBOSE", "").lower() in ("1", "true", "yes"),
        )


class CertificateConfig(BaseModel):
    """Configuration for certificate generation and verification."""

    # Spectral analysis settings
    default_rank_k: int = Field(default=10, ge=1, description="Default rank for spectral analysis")
    min_trace_length: int = Field(default=5, ge=2, description="Minimum trace length for analysis")

    # Threshold settings
    residual_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Threshold for residual-based drift detection"
    )
    auc_threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Minimum AUC for classifier validation"
    )
    tpr_at_fpr05_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum TPR at FPR=0.05"
    )

    # Numerical stability thresholds
    witness_condition_number_threshold: float = Field(
        default=1e8,
        ge=1.0,
        description=(
            "Maximum allowed condition number for witness matrices. "
            "Rationale: Condition numbers > 1e8 indicate near-singularity and "
            "can lead to unreliable numerical results. This threshold provides "
            "a conservative guard against ill-conditioned systems."
        )
    )
    witness_gap_threshold: float = Field(
        default=1e-6,
        ge=0.0,
        description=(
            "Minimum singular value gap for witness validation. "
            "Rationale: Gaps < 1e-6 indicate near-degenerate subspaces that "
            "may not be well-separated under perturbation (Wedin's Theorem). "
            "This ensures robust subspace identification."
        )
    )
    explained_variance_threshold: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum explained variance for PCA rank selection. "
            "Rationale: 90% explained variance ensures we capture dominant "
            "dynamics while avoiding overfitting. Validated empirically on "
            "typical agent trajectories with 128-384 dimensional embeddings."
        )
    )
    oos_validation_k: int = Field(
        default=3,
        ge=1,
        description=(
            "Number of out-of-sample steps for cross-validation. "
            "Rationale: K=3 provides sufficient holdout data for OOS residual "
            "estimation while preserving enough training data. For traces with "
            "T < 12 steps, K is adaptively reduced to max(1, T // 4)."
        )
    )
    oos_residual_floor: float = Field(
        default=0.1,
        ge=0.0,
        description=(
            "Conservative residual floor for degenerate/short traces. "
            "Rationale: Returning 0.0 residual on traces with T < 4 steps "
            "produces overly optimistic certificates. A floor of 0.1 indicates "
            "'minimal but uncertain' predictability, preventing zero-residual "
            "pathology on traces too short for meaningful cross-validation."
        )
    )

    # Kernel settings
    verified_kernel_path: Optional[str] = Field(
        default=None,
        description="Path to verified kernel for reproducible analysis"
    )
    allow_kernel_load: bool = Field(
        default=True,
        description="Allow loading verified kernel"
    )
    skip_verified_kernel: bool = Field(
        default=False,
        description="Skip verified kernel and use Python fallback"
    )
    kernel_service_mode: bool = Field(
        default=False,
        description="Use kernel as a persistent service"
    )
    kernel_image: str = Field(
        default="ipsissima/kernel:latest",
        description="Docker image for kernel execution"
    )
    kernel_local_py: str = Field(
        default="",
        description="Local Python script path for kernel (for testing)"
    )
    kernel_timeout: int = Field(
        default=300,
        ge=1,
        description="Kernel execution timeout in seconds"
    )
    kernel_socket: str = Field(
        default="/tmp/esm_kernel.sock",
        description="Socket path for kernel service"
    )
    generation_timeout: int = Field(
        default=300,
        ge=1,
        description="Agent generation timeout in seconds"
    )

    @classmethod
    def from_env(cls) -> "CertificateConfig":
        """Load certificate configuration from environment variables.
        
        Environment variables:
            SPECTRAL_RANK_K: Default rank for spectral analysis (default: 10)
            MIN_TRACE_LENGTH: Minimum trace length (default: 5)
            RESIDUAL_THRESHOLD: Residual threshold for drift detection (default: 0.3)
            AUC_THRESHOLD: Minimum AUC for classifier validation (default: 0.90)
            TPR_THRESHOLD: Minimum TPR at FPR=0.05 (default: 0.80)
            WITNESS_COND_THRESHOLD: Witness condition number threshold (default: 1e8)
            WITNESS_GAP_THRESHOLD: Witness singular value gap threshold (default: 1e-6)
            EXPLAINED_VARIANCE_THRESHOLD: PCA explained variance threshold (default: 0.90)
            OOS_VALIDATION_K: Out-of-sample validation steps (default: 3)
            OOS_RESIDUAL_FLOOR: Conservative residual floor (default: 0.1)
            VERIFIED_KERNEL_PATH: Path to verified kernel
            ESM_ALLOW_KERNEL_LOAD: Allow kernel loading (default: 1)
            ESM_SKIP_VERIFIED_KERNEL: Skip verified kernel (default: 0)
            ESM_KERNEL_SERVICE: Use kernel as service (default: 0)
            ESM_KERNEL_IMAGE: Docker image for kernel (default: ipsissima/kernel:latest)
            ESM_KERNEL_LOCAL_PY: Local Python kernel script path
            ESM_KERNEL_TIMEOUT: Kernel timeout in seconds (default: 300)
            ESM_KERNEL_SOCKET: Kernel service socket path (default: /tmp/esm_kernel.sock)
            ESM_GEN_TIMEOUT: Generation timeout in seconds (default: 300)
        """
        return cls(
            default_rank_k=int(os.environ.get("SPECTRAL_RANK_K", "10")),
            min_trace_length=int(os.environ.get("MIN_TRACE_LENGTH", "5")),
            residual_threshold=float(os.environ.get("RESIDUAL_THRESHOLD", "0.3")),
            auc_threshold=float(os.environ.get("AUC_THRESHOLD", "0.90")),
            tpr_at_fpr05_threshold=float(os.environ.get("TPR_THRESHOLD", "0.80")),
            witness_condition_number_threshold=float(os.environ.get("WITNESS_COND_THRESHOLD", "1e8")),
            witness_gap_threshold=float(os.environ.get("WITNESS_GAP_THRESHOLD", "1e-6")),
            explained_variance_threshold=float(os.environ.get("EXPLAINED_VARIANCE_THRESHOLD", "0.90")),
            oos_validation_k=int(os.environ.get("OOS_VALIDATION_K", "3")),
            oos_residual_floor=float(os.environ.get("OOS_RESIDUAL_FLOOR", "0.1")),
            verified_kernel_path=os.environ.get("VERIFIED_KERNEL_PATH"),
            allow_kernel_load=os.environ.get("ESM_ALLOW_KERNEL_LOAD", "1") in ("1", "true", "yes"),
            skip_verified_kernel=os.environ.get("ESM_SKIP_VERIFIED_KERNEL", "").lower() in ("1", "true", "yes"),
            kernel_service_mode=os.environ.get("ESM_KERNEL_SERVICE", "0") == "1",
            kernel_image=os.environ.get("ESM_KERNEL_IMAGE", "ipsissima/kernel:latest"),
            kernel_local_py=os.environ.get("ESM_KERNEL_LOCAL_PY", ""),
            kernel_timeout=int(os.environ.get("ESM_KERNEL_TIMEOUT", "300")),
            kernel_socket=os.environ.get("ESM_KERNEL_SOCKET", "/tmp/esm_kernel.sock"),
            generation_timeout=int(os.environ.get("ESM_GEN_TIMEOUT", "300")),
        )


# Global config instances (lazily loaded)
_assessor_config: Optional[AssessorConfig] = None
_certificate_config: Optional[CertificateConfig] = None


def get_assessor_config() -> AssessorConfig:
    """Get the global assessor configuration, loading from environment if needed."""
    global _assessor_config
    if _assessor_config is None:
        _assessor_config = AssessorConfig.from_env()
    return _assessor_config


def get_certificate_config() -> CertificateConfig:
    """Get the global certificate configuration, loading from environment if needed."""
    global _certificate_config
    if _certificate_config is None:
        _certificate_config = CertificateConfig.from_env()
    return _certificate_config


def reset_config() -> None:
    """Reset global config instances (useful for testing)."""
    global _assessor_config, _certificate_config
    _assessor_config = None
    _certificate_config = None


__all__ = [
    "AssessorConfig",
    "CertificateConfig",
    "get_assessor_config",
    "get_certificate_config",
    "reset_config",
]
