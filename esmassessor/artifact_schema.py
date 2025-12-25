from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CertifiedVerdict(str, Enum):
    """Verdict for hybrid validation combining spectral stability and execution success."""
    PASS = "PASS"
    FAIL_DRIFT = "FAIL_DRIFT"
    FAIL_EXECUTION = "FAIL_EXECUTION"
    FAIL_ROBUSTNESS = "FAIL_ROBUSTNESS"
    FAIL_SEMANTIC = "FAIL_SEMANTIC"


class SpectralMetrics(BaseModel):
    pca_explained: float = Field(..., description="Explained variance captured by PCA")
    max_eig: float = Field(..., description="Maximum eigenvalue magnitude")
    spectral_gap: float = Field(..., description="Gap between leading Koopman eigenvalues")
    residual: float = Field(..., description="Koopman residual ratio")
    pca_tail_estimate: float = Field(..., description="Estimated PCA tail mass")
    semantic_divergence: float = Field(0.0, description="Mean cosine distance from task embedding")
    theoretical_bound: float = Field(..., description="Residual + tail + semantic upper bound")
    task_score: Optional[float] = Field(None, description="Episode score or heuristic reward")
    trace_path: Optional[str] = Field(None, description="Path to saved trace JSON")


class ExecutionWitness(BaseModel):
    """Execution evidence from running agent-generated tests and ground truth tests."""
    ground_truth_passed: bool = Field(..., description="Whether ground truth tests passed")
    agent_generated_passed: Optional[bool] = Field(
        None, description="Whether agent-generated verify() block passed (if provided)"
    )
    agent_verify_block: Optional[str] = Field(None, description="The extracted verify() block code")
    execution_log: Optional[str] = Field(None, description="Captured output/errors from test execution")
    semantic_oracle_passed: bool = Field(True, description="Whether semantic oracle checks passed")


class HybridCertificate(BaseModel):
    """Hybrid validity proof combining spectral stability and execution success."""
    spectral_metrics: SpectralMetrics = Field(..., description="Spectral stability metrics")
    execution_verified: bool = Field(..., description="True only if execution tests passed")
    semantic_compliance: float = Field(
        ..., description="Score from semantic oracle (0.0-1.0, higher is better)"
    )
    certified_verdict: CertifiedVerdict = Field(
        ..., description="Final verdict: PASS | FAIL_DRIFT | FAIL_EXECUTION | FAIL_ROBUSTNESS | FAIL_SEMANTIC"
    )
    execution_witness: Optional[ExecutionWitness] = Field(
        None, description="Evidence from proof-carrying code execution"
    )
    theoretical_bound: float = Field(..., description="Spectral stability bound (lower is better)")
    reasoning: Optional[str] = Field(
        None, description="Human-readable explanation of the verdict"
    )

    class Config:
        extra = "allow"


class CertificateArtifact(BaseModel):
    episode: str
    episode_id: str
    participant: str
    spectral_metrics: SpectralMetrics

    class Config:
        extra = "allow"


__all__ = ["SpectralMetrics", "CertificateArtifact", "HybridCertificate", "ExecutionWitness", "CertifiedVerdict"]
