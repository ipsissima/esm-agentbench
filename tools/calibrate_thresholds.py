#!/usr/bin/env python3
"""Calibrate residual and jump thresholds for the ESM assessor."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
TUTORIAL_SRC = PROJECT_ROOT / "third_party" / "tutorial" / "src"
if str(TUTORIAL_SRC) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_SRC))

from certificates.make_certificate import compute_certificate
from third_party.tutorial.src.agentbeats.runner import _synthetic_trace

LOGGER = logging.getLogger(__name__)


def load_config(path: Path) -> Dict[str, Any]:
    import yaml

    if not path.exists():
        raise RuntimeError(f"config missing: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def embed_texts(texts: Iterable[str], backend: str, config: Dict[str, Any]) -> np.ndarray:
    if os.environ.get("ESM_FORCE_TFIDF"):
        backend = "tfidf"
    backend = backend or (config.get("canonical", {}) or {}).get("embedding_backend", "sentence-transformers")
    if backend == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer

            model_name = (config.get("canonical", {}) or {}).get("sentence_transformers_model", "all-MiniLM-L6-v2")
            model = SentenceTransformer(model_name)
            arr = np.asarray(model.encode(list(texts), normalize_embeddings=True), dtype=float)
            return arr
        except Exception as exc:  # pragma: no cover - fallback
            LOGGER.warning("sentence-transformers unavailable: %s; falling back to tf-idf", exc)
            backend = "tfidf"
    if backend == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer

        vect = TfidfVectorizer(ngram_range=(1, 2), max_features=(config.get("canonical", {}) or {}).get("tfidf_max_features", 512))
        mat = vect.fit_transform(list(texts)).toarray().astype(float)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        return mat / norms
    raise RuntimeError(f"unsupported backend {backend}")


def compute_residuals(trials: int, prompt: str, backend: str, config: Dict[str, Any], seed: int, bad: bool = False) -> List[float]:
    rng = np.random.default_rng(seed + (1 if bad else 0))
    residuals: List[float] = []
    for idx in range(trials):
        trace = _synthetic_trace("debater" if not bad else "noisy", f"{prompt} #{idx}", max_steps=5, seed=seed + idx)
        if bad:
            for step in trace:
                step["text"] += " noisy pivot" * 2
        embeddings = embed_texts([s["text"] for s in trace], backend, config)
        cert = compute_certificate(embeddings, r=(config.get("certificate", {}) or {}).get("pca_rank", 10))
        residuals.append(float(cert.get("residual", 1.0)))
    return residuals


def calibrate(backend: str, trials: int, config: Dict[str, Any], seed: int, dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        rng = np.random.default_rng(seed)
        residuals_good = rng.uniform(0.05, 0.35, size=trials).tolist()
        residuals_bad = rng.uniform(0.55, 0.95, size=trials).tolist()
    else:
        residuals_good = compute_residuals(trials, "coherent argument", backend, config, seed, bad=False)
        residuals_bad = compute_residuals(trials, "chaotic rebuttal", backend, config, seed, bad=True)

    all_vals = np.array(residuals_good + residuals_bad)
    thresholds = np.linspace(float(np.min(all_vals)), float(np.max(all_vals)), num=100)
    best_thresh = thresholds[0]
    best_score = -1.0
    for t in thresholds:
        preds = np.array([1 if r <= t else 0 for r in all_vals])
        labels = np.array([1] * len(residuals_good) + [0] * len(residuals_bad))
        tp = float(np.sum((preds == 1) & (labels == 1)))
        fp = float(np.sum((preds == 1) & (labels == 0)))
        fn = float(np.sum((preds == 0) & (labels == 1)))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        if f1 > best_score:
            best_score = f1
            best_thresh = float(t)
    jump_factor = float(np.percentile(residuals_bad, 90) / max(np.percentile(residuals_good, 10), 1e-6))
    return {
        "backend": backend,
        "threshold": best_thresh,
        "threshold_f1": best_score,
        "jump_factor": jump_factor,
        "residuals_good": residuals_good,
        "residuals_bad": residuals_bad,
    }


def update_config(cfg_path: Path, result: Dict[str, Any]) -> None:
    import yaml

    cfg = load_config(cfg_path)
    cal = cfg.get("calibration", {}) or {}
    cal["residual_threshold"] = float(result["threshold"])
    cal["jump_factor"] = float(result["jump_factor"])
    cal["calibrated_on"] = f"{result['backend']}:demo"
    cfg["calibration"] = cal
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate thresholds using synthetic debate traces")
    parser.add_argument("--backend", default="sentence-transformers")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-cache-only", action="store_true", help="reuse existing calibration file if present")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg_path = PROJECT_ROOT / "evaluation_config.yaml"
    config = load_config(cfg_path)

    calib_path = PROJECT_ROOT / "certificates" / f"calibration_{args.backend}.json"
    if args.use_cache_only and calib_path.exists():
        LOGGER.info("Using cached calibration at %s", calib_path)
        return

    calib_result = calibrate(args.backend, args.trials, config, args.seed, dry_run=args.dry_run)
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    calib_path.write_text(json.dumps(calib_result, indent=2), encoding="utf-8")
    update_config(cfg_path, calib_result)
    LOGGER.info("wrote calibration to %s", calib_path)


if __name__ == "__main__":  # pragma: no cover
    main()
