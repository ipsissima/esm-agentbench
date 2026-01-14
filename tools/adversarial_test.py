#!/usr/bin/env python3
"""Adversarial testing for drift detection classifier on embeddings."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import importlib.util
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from tools.feature_utils import (
    NormalizationConfig,
    PROJECT_ROOT,
    compute_trace_features,
    infer_label,
    label_to_binary,
    load_trace_json,
)
from tools.eval_holdout import evaluate_scores

logger = logging.getLogger(__name__)

HAS_CMA = importlib.util.find_spec("cma") is not None
if HAS_CMA:
    import cma


def _load_model(model_path: Path) -> Tuple[Any, List[str]]:
    payload = joblib.load(model_path)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"], payload.get("feature_columns", [])
    return payload, []


def _trace_files(traces_dir: Path) -> List[Path]:
    if traces_dir.is_file():
        return [traces_dir]
    return sorted(traces_dir.glob("**/*.json"))


def _compute_feature_vector(
    embeddings: np.ndarray,
    feature_cols: List[str],
    normalization: NormalizationConfig,
    trim_proportion: float = 0.0,
) -> np.ndarray:
    normalization.trim_proportion = trim_proportion
    features = compute_trace_features(embeddings, normalization=normalization, k=10, kernel_strict=False)
    row = np.array([features.get(col, 0.0) for col in feature_cols], dtype=np.float64)
    return row.reshape(1, -1)


def _score_trace(
    model: Any,
    embeddings: np.ndarray,
    feature_cols: List[str],
    normalization: NormalizationConfig,
    trim_proportion: float = 0.0,
) -> float:
    X = _compute_feature_vector(embeddings, feature_cols, normalization, trim_proportion)
    return float(model.predict_proba(X)[0, 1])


def _apply_delta(embeddings: np.ndarray, delta: np.ndarray, norm_clip: float) -> np.ndarray:
    perturbed = embeddings + delta
    if norm_clip <= 0:
        return perturbed
    norms = np.linalg.norm(perturbed - embeddings, axis=1, keepdims=True)
    scale = np.minimum(1.0, norm_clip / (norms + 1e-12))
    return embeddings + (perturbed - embeddings) * scale


def _cmaes_attack(
    embeddings: np.ndarray,
    model: Any,
    feature_cols: List[str],
    normalization: NormalizationConfig,
    budget: int,
    norm_clip: float,
    seed: int,
) -> Tuple[np.ndarray, float]:
    d = embeddings.shape[1]
    rng = np.random.default_rng(seed)

    def objective(delta_flat: np.ndarray) -> float:
        delta = np.tile(delta_flat.reshape(1, -1), (embeddings.shape[0], 1))
        perturbed = _apply_delta(embeddings, delta, norm_clip)
        return _score_trace(model, perturbed, feature_cols, normalization)

    if HAS_CMA:
        es = cma.CMAEvolutionStrategy(np.zeros(d), 0.1, {"seed": seed, "maxfevals": budget})
        while not es.stop():
            solutions = es.ask()
            scores = [objective(np.array(s)) for s in solutions]
            es.tell(solutions, scores)
            if es.countevals >= budget:
                break
        best = np.array(es.best.x)
        final_score = objective(best)
        delta = np.tile(best.reshape(1, -1), (embeddings.shape[0], 1))
        return _apply_delta(embeddings, delta, norm_clip), float(final_score)

    best_delta = rng.normal(scale=0.1, size=d)
    best_score = objective(best_delta)
    for _ in range(budget):
        candidate = rng.normal(scale=0.1, size=d)
        score = objective(candidate)
        if score < best_score:
            best_score = score
            best_delta = candidate
    delta = np.tile(best_delta.reshape(1, -1), (embeddings.shape[0], 1))
    return _apply_delta(embeddings, delta, norm_clip), float(best_score)


def _finite_diff_attack(
    embeddings: np.ndarray,
    model: Any,
    feature_cols: List[str],
    normalization: NormalizationConfig,
    budget: int,
    norm_clip: float,
    seed: int,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    d = embeddings.shape[1]
    delta = np.zeros(d, dtype=np.float64)
    eps = 1e-2
    step = 0.1

    for _ in range(budget):
        direction = rng.normal(size=d)
        direction /= np.linalg.norm(direction) + 1e-12
        score_plus = _score_trace(
            model,
            _apply_delta(embeddings, np.tile((delta + eps * direction).reshape(1, -1), (embeddings.shape[0], 1)), norm_clip),
            feature_cols,
            normalization,
        )
        score_minus = _score_trace(
            model,
            _apply_delta(embeddings, np.tile((delta - eps * direction).reshape(1, -1), (embeddings.shape[0], 1)), norm_clip),
            feature_cols,
            normalization,
        )
        grad = (score_plus - score_minus) / (2 * eps)
        delta = delta - step * grad * direction
        if norm_clip > 0:
            norm = np.linalg.norm(delta)
            if norm > norm_clip:
                delta = delta / norm * norm_clip

    perturbed = _apply_delta(embeddings, np.tile(delta.reshape(1, -1), (embeddings.shape[0], 1)), norm_clip)
    final_score = _score_trace(model, perturbed, feature_cols, normalization)
    return perturbed, float(final_score)


def _ensemble_defense_score(
    model: Any,
    embeddings: np.ndarray,
    feature_cols: List[str],
    normalization: NormalizationConfig,
) -> float:
    base_score = _score_trace(model, embeddings, feature_cols, normalization)
    alt_norm = NormalizationConfig(
        l2_normalize_steps=True,
        zscore_per_trace=True,
        length_normalize=normalization.length_normalize,
        trim_proportion=normalization.trim_proportion,
    )
    alt_score = _score_trace(model, embeddings, feature_cols, alt_norm)
    return float(0.5 * (base_score + alt_score))


def _robust_defense_score(
    model: Any,
    embeddings: np.ndarray,
    feature_cols: List[str],
    normalization: NormalizationConfig,
) -> float:
    return _score_trace(model, embeddings, feature_cols, normalization, trim_proportion=0.1)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Run adversarial tests against tuned classifier.")
    parser.add_argument("--traces-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--attack", choices=["cmaes", "finite-diff"], default="cmaes")
    parser.add_argument("--budget", type=int, default=200)
    parser.add_argument("--norm-clip", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    model, feature_cols = _load_model(args.model)
    if not feature_cols:
        raise ValueError("Model metadata missing feature columns")

    normalization = NormalizationConfig()
    trace_paths = _trace_files(args.traces_dir)
    if not trace_paths:
        raise FileNotFoundError("No trace JSON files found for adversarial testing")

    results = []
    y_true = []
    base_scores = []
    attacked_scores = []
    ensemble_scores = []
    robust_scores = []

    for idx, trace_path in enumerate(trace_paths):
        trace_data = load_trace_json(trace_path)
        embeddings = trace_data.get("embeddings")
        if embeddings is None:
            logger.warning("Skipping %s: no embeddings", trace_path)
            continue
        embeddings = np.asarray(embeddings, dtype=np.float64)
        label = infer_label(trace_data.get("label", trace_path.stem))
        label_bin = label_to_binary(label)

        base_score = _score_trace(model, embeddings, feature_cols, normalization)

        if args.attack == "cmaes":
            perturbed, attacked_score = _cmaes_attack(
                embeddings,
                model,
                feature_cols,
                normalization,
                budget=args.budget,
                norm_clip=args.norm_clip,
                seed=args.seed + idx,
            )
        else:
            perturbed, attacked_score = _finite_diff_attack(
                embeddings,
                model,
                feature_cols,
                normalization,
                budget=args.budget,
                norm_clip=args.norm_clip,
                seed=args.seed + idx,
            )

        ensemble_score = _ensemble_defense_score(model, perturbed, feature_cols, normalization)
        robust_score = _robust_defense_score(model, perturbed, feature_cols, normalization)

        results.append({
            "trace": str(trace_path),
            "label": label,
            "base_score": base_score,
            "attacked_score": attacked_score,
            "ensemble_score": ensemble_score,
            "robust_score": robust_score,
        })

        if label_bin is not None:
            y_true.append(label_bin)
            base_scores.append(base_score)
            attacked_scores.append(attacked_score)
            ensemble_scores.append(ensemble_score)
            robust_scores.append(robust_score)

    metrics = {}
    if len(set(y_true)) > 1:
        y_true_arr = np.array(y_true)
        metrics["baseline"] = evaluate_scores(
            y_true_arr,
            np.array(base_scores),
            fpr_target=0.05,
            n_boot=200,
            seed=args.seed,
        )
        metrics["attacked"] = evaluate_scores(
            y_true_arr,
            np.array(attacked_scores),
            fpr_target=0.05,
            n_boot=200,
            seed=args.seed,
        )
        metrics["ensemble_defense"] = evaluate_scores(
            y_true_arr,
            np.array(ensemble_scores),
            fpr_target=0.05,
            n_boot=200,
            seed=args.seed,
        )
        metrics["robust_defense"] = evaluate_scores(
            y_true_arr,
            np.array(robust_scores),
            fpr_target=0.05,
            n_boot=200,
            seed=args.seed,
        )
    else:
        metrics["note"] = "Insufficient label diversity for AUC/TPR"

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_path = reports_dir / "adversarial_logs.jsonl"
    with open(log_path, "w", encoding="utf-8") as handle:
        for entry in results:
            handle.write(json.dumps(entry) + "\n")
    logger.info("Wrote adversarial logs to %s", log_path)

    summary = {
        "attack": args.attack,
        "budget": args.budget,
        "norm_clip": args.norm_clip,
        "has_cma": HAS_CMA,
        "metrics": metrics,
    }

    summary_path = reports_dir / "adversarial_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Saved adversarial summary to %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
