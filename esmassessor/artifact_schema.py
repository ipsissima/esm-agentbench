from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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


class CertificateArtifact(BaseModel):
    episode: str
    episode_id: str
    participant: str
    spectral_metrics: SpectralMetrics

    class Config:
        extra = "allow"


__all__ = ["SpectralMetrics", "CertificateArtifact"]
