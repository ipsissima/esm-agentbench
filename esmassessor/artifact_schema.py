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


class MultiScaleVerdict(str, Enum):
    """Verdict for multi-scale spectral monitoring (Phase 4 AgentX).

    The dual-metric approach handles context switches without false positives
    while catching slow drift vulnerabilities.
    """
    PASS = "PASS"  # Local stable AND Global aligned
    FAIL_INSTABILITY = "FAIL_INSTABILITY"  # Local chaotic (high spectral bound)
    FAIL_GOAL_DRIFT = "FAIL_GOAL_DRIFT"  # Local stable BUT Global divergent


class SpectralMetrics(BaseModel):
    pca_explained: float = Field(..., description="Explained variance captured by SVD truncation")
    sigma_max: float = Field(..., description="Leading singular value (stability proxy via Wedin's theorem)")
    singular_gap: float = Field(..., description="Gap between leading singular values (Wedin margin)")
    residual: float = Field(..., description="Trajectory prediction residual (||X1 - A X0|| / ||X1||)")
    tail_energy: float = Field(..., description="Energy lost in rank truncation (unexplained variance)")
    semantic_divergence: float = Field(0.0, description="Mean cosine distance from task embedding")
    theoretical_bound: float = Field(..., description="SVD-based stability bound (residual + tail + semantic)")
    task_score: Optional[float] = Field(None, description="Episode score or heuristic reward")
    trace_path: Optional[str] = Field(None, description="Path to saved trace JSON")


class LocalCoherenceMetrics(BaseModel):
    """Micro-Monitor: Local Coherence metrics from sliding window SVD.

    Measures sub-task consistency: "Is the agent executing the current sub-task consistently?"
    """
    local_spectral_bound: float = Field(..., description="Spectral bound for current window (lower is better)")
    local_residual: float = Field(..., description="Prediction residual within window")
    local_pca_explained: float = Field(..., description="Variance explained within window")
    local_tail_energy: float = Field(..., description="Unexplained energy in window")
    window_start: int = Field(..., description="Start index of analyzed window")
    window_end: int = Field(..., description="End index of analyzed window")


class GlobalAlignmentMetrics(BaseModel):
    """Macro-Monitor: Global Alignment metrics from semantic divergence.

    Measures goal alignment: "Has the agent forgotten the user's intent?"
    """
    global_semantic_drift: float = Field(..., description="Weighted semantic drift (higher = more drift)")
    window_semantic_drift: float = Field(..., description="Drift of current window from task")
    cumulative_semantic_drift: float = Field(..., description="Drift across entire trace")
    max_semantic_drift: float = Field(..., description="Maximum drift at any single step")


class ContextSegment(BaseModel):
    """A detected context segment in the trace (e.g., Planning, Coding, Testing)."""
    start: int = Field(..., description="Start index of segment")
    end: int = Field(..., description="End index of segment")
    length: int = Field(..., description="Length of segment (end - start)")


class MultiScaleCertificate(BaseModel):
    """Phase 4 (AgentX): Adaptive Multi-Scale Spectral Certificate.

    Handles long-horizon tasks with context switches (Planning → Coding → Testing)
    using dual-metric verification:
    - Local Coherence: Is the current sub-task stable?
    - Global Alignment: Is the agent still aligned with the original goal?

    The verdict logic:
    - PASS: Local stable AND Global aligned
    - FAIL_INSTABILITY: Local chaotic (high spectral bound in current segment)
    - FAIL_GOAL_DRIFT: Local stable BUT Global divergent (agent forgot the mission)
    """
    multi_scale_verdict: MultiScaleVerdict = Field(..., description="Dual-metric verdict")
    local_spectral_bound: float = Field(..., description="Spectral stability of current segment/window")
    global_semantic_drift: float = Field(..., description="Semantic alignment with task embedding")
    local_coherence: LocalCoherenceMetrics = Field(..., description="Full local coherence metrics")
    global_alignment: GlobalAlignmentMetrics = Field(..., description="Full global alignment metrics")
    segments: List[ContextSegment] = Field(default_factory=list, description="Detected context segments")
    active_segment: Optional[ContextSegment] = Field(None, description="Currently active segment")
    num_context_switches: int = Field(0, description="Number of detected context switches")
    # Base certificate metrics for compatibility
    theoretical_bound: float = Field(..., description="Overall spectral stability bound")
    residual: float = Field(..., description="Overall prediction residual")
    tail_energy: float = Field(..., description="Overall unexplained energy")
    pca_explained: float = Field(..., description="Overall variance explained")
    semantic_divergence: float = Field(..., description="Overall semantic divergence")
    reasoning: Optional[str] = Field(None, description="Human-readable explanation of verdict")

    class Config:
        extra = "allow"


class ExecutionWitness(BaseModel):
    """Execution evidence from running agent-generated tests and ground truth tests."""
    ground_truth_passed: bool = Field(..., description="Whether ground truth tests passed")
    agent_generated_passed: Optional[bool] = Field(
        None, description="Whether agent-generated verify() block passed (if provided)"
    )
    agent_verify_block: Optional[str] = Field(None, description="The extracted verify() block code")
    execution_log: Optional[str] = Field(None, description="Captured output/errors from test execution")
    semantic_oracle_passed: bool = Field(True, description="Whether semantic oracle checks passed")


class CertificateAudit(BaseModel):
    """Audit metadata for certificate provenance."""
    witness_hash: str = Field(..., description="SHA256 hash of the input trajectory")
    embedder_id: str = Field(..., description="Identifier for the embedding model")
    numerical_diagnostics: Dict[str, float] = Field(
        default_factory=dict,
        description="Numerical diagnostics such as condition numbers and gaps",
    )
    kernel_mode: str = Field(..., description="Kernel execution mode")
    timestamp_utc: str = Field(..., description="UTC timestamp when the certificate was generated")


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
    audit: Optional[CertificateAudit] = Field(
        None, description="Audit provenance metadata for the certificate"
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


__all__ = [
    "SpectralMetrics",
    "CertificateArtifact",
    "HybridCertificate",
    "ExecutionWitness",
    "CertificateAudit",
    "CertifiedVerdict",
    # Phase 4: Multi-Scale Spectral Monitoring
    "MultiScaleVerdict",
    "LocalCoherenceMetrics",
    "GlobalAlignmentMetrics",
    "ContextSegment",
    "MultiScaleCertificate",
]
