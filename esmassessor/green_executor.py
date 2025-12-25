"""Green executor that computes spectral certificates for AgentBeats traces."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from esmassessor.artifacts import send_task_update
from esmassessor.base import Assessment, GreenExecutorBase
from certificates.make_certificate import compute_certificate
from esmassessor.artifact_schema import CertificateArtifact, SpectralMetrics
from esmassessor.write_artifact import write_certificate_artifact

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class EsmGreenExecutor(GreenExecutorBase):
    """Green executor computing spectral certificates for traces."""

    def __init__(self, evaluation_config: Path | str | None = None, assessment_timeout: float = 120.0) -> None:
        super().__init__(assessment_timeout=assessment_timeout)
        self.config_path = Path(evaluation_config or PROJECT_ROOT / "evaluation_config.yaml")
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        import yaml

        if not self.config_path.exists():
            raise RuntimeError(f"evaluation_config missing at {self.config_path}")
        return yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

    def _guard_patent_safety(self, assessment: Assessment) -> None:
        if "fragment" in assessment.prompt.lower() or "colimit" in assessment.prompt.lower():
            raise RuntimeError("Fragment/colimit/compilation disallowed in public ESM")

    def _embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        backend = os.environ.get("ESM_FORCE_TFIDF") and "tfidf" or (
            (self.config.get("canonical", {}) or {}).get("embedding_backend", "sentence-transformers")
        )
        model_name = (self.config.get("canonical", {}) or {}).get("sentence_transformers_model", "all-MiniLM-L6-v2")
        if backend == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(model_name)
                vecs = model.encode(list(texts), normalize_embeddings=True)
                arr = np.asarray(vecs, dtype=float)
                if arr.ndim == 1:
                    arr = arr.reshape(len(list(texts)), -1)
                return arr
            except Exception as exc:  # pragma: no cover - fallback path
                LOGGER.warning("sentence-transformers unavailable: %s; falling back to tf-idf", exc)
                backend = "tfidf"
        if backend == "tfidf":
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), max_features=self.config.get("canonical", {}).get("tfidf_max_features", 512)
            )
            mat = vectorizer.fit_transform(list(texts)).toarray().astype(float)
            norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
            return mat / norms
        raise RuntimeError(f"Unsupported embedding backend: {backend}")

    def _certificate_for_trace(self, assessment: Assessment, participant: str, trace: List[Mapping[str, Any]]) -> CertificateArtifact:
        texts = [str(step.get("text", "")) for step in trace]
        embeddings = self._embed_texts(texts) if texts else np.zeros((0, 1))
        pca_rank = int((self.config.get("certificate", {}) or {}).get("pca_rank", 10))

        # Extract task embedding for semantic divergence computation
        # Priority: assessment.prompt > first step text
        task_embedding = None
        if assessment.prompt:
            # Embed the task/prompt for semantic divergence anchor
            task_embeddings = self._embed_texts([assessment.prompt])
            if task_embeddings.shape[0] > 0:
                task_embedding = task_embeddings[0]
        elif embeddings.shape[0] > 0:
            # Fallback: use first step embedding as task reference
            task_embedding = embeddings[0]

        cert_data = compute_certificate(embeddings, r=pca_rank, task_embedding=task_embedding)
        score = float(np.mean([float(s.get("confidence", 0.5)) for s in trace])) if trace else None
        metrics = SpectralMetrics(
            pca_explained=float(cert_data.get("pca_explained", 0.0)),
            sigma_max=float(cert_data.get("sigma_max", 0.0)),
            singular_gap=float(cert_data.get("singular_gap", 0.0)),
            residual=float(cert_data.get("residual", 0.0)),
            tail_energy=float(cert_data.get("tail_energy", 0.0)),
            semantic_divergence=float(cert_data.get("semantic_divergence", 0.0)),
            theoretical_bound=float(cert_data.get("theoretical_bound", 0.0)),
            task_score=score,
            trace_path=None,
        )
        return CertificateArtifact(
            episode=assessment.prompt,
            episode_id=assessment.assessment_id,
            participant=participant,
            spectral_metrics=metrics,
        )

    def assess(self, assessment: Assessment, traces: Mapping[str, List[Mapping[str, Any]]]) -> Dict[str, Any]:
        self._guard_patent_safety(assessment)
        self.on_start(assessment)
        outdir = Path("demo_traces")
        outdir.mkdir(parents=True, exist_ok=True)
        results: Dict[str, Any] = {}
        for participant, trace in traces.items():
            certificate = self._certificate_for_trace(assessment, participant, trace)
            cert_path = outdir / f"{assessment.assessment_id}_{participant}_certificate.json"
            trace_path = outdir / f"{assessment.assessment_id}_{participant}_trace.json"
            trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
            cert_dict = certificate.dict()
            cert_dict["spectral_metrics"]["trace_path"] = str(trace_path)
            write_certificate_artifact(cert_dict, cert_path)
            results[participant] = cert_dict
            send_task_update({"participant": participant, "certificate": cert_dict})
        self.on_complete(assessment, results)
        return results


__all__ = ["EsmGreenExecutor"]
