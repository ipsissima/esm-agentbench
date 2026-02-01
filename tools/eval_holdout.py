#!/usr/bin/env python3
"""Evaluate calibrated classifier on holdout set with bootstrap CIs."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.feature_utils import PROJECT_ROOT, label_to_binary

logger = logging.getLogger(__name__)


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> Tuple[float, float, float]:
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0, 1.0
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0, float(fpr[0]) if len(fpr) else 1.0, float(thresholds[0]) if len(thresholds) else 0.5
    best = idx[-1]
    return float(tpr[best]), float(fpr[best]), float(thresholds[best])


def bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))
    if not aucs:
        return 0.5, 0.5
    low, high = np.percentile(aucs, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_tpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_fpr: float,
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    tprs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        tpr, _, _ = tpr_at_fpr(y_true[idx], y_score[idx], target_fpr)
        tprs.append(tpr)
    low, high = np.percentile(tprs, [2.5, 97.5])
    return float(low), float(high)


def load_model(model_path: Path) -> Tuple[Any, list, float, float]:
    payload = joblib.load(model_path)
    if isinstance(payload, dict) and "model" in payload:
        return (
            payload["model"],
            payload.get("feature_columns", []),
            payload.get("threshold", 0.5),
            payload.get("fpr_target", 0.05),
        )
    return payload, [], 0.5, 0.05


def evaluate_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fpr_target: float,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    auc_score = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5
    tpr, fpr, threshold = tpr_at_fpr(y_true, y_score, fpr_target)

    auc_ci = bootstrap_auc(y_true, y_score, n_boot=n_boot, seed=seed)
    tpr_ci = bootstrap_tpr(y_true, y_score, fpr_target, n_boot=n_boot, seed=seed)

    preds = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "auc": float(auc_score),
        "auc_ci": {"low": auc_ci[0], "high": auc_ci[1]},
        "tpr_at_fpr": float(tpr),
        "tpr_ci": {"low": tpr_ci[0], "high": tpr_ci[1]},
        "fpr": float(fpr),
        "threshold": float(threshold),
        "confusion": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "tpr": float(tp / max(tp + fn, 1)),
            "fpr": float(fp / max(fp + tn, 1)),
        },
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate tuned classifier on holdout features.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--fpr-target", type=float, default=0.05)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    model, feature_cols, model_threshold, model_fpr = load_model(args.model)
    df = pd.read_csv(args.features_csv)

    if "label" not in df.columns:
        raise ValueError("features CSV must include label column")

    df = df.copy()
    df["label_binary"] = df["label"].apply(label_to_binary)
    df = df.dropna(subset=["label_binary"])
    y_true = df["label_binary"].astype(int).to_numpy()

    if not feature_cols:
        feature_cols = [col for col in df.columns if col not in {"label", "label_binary", "scenario", "run_id"}]
    X = df[feature_cols].fillna(0.0)

    y_score = model.predict_proba(X)[:, 1]

    metrics = evaluate_scores(
        y_true,
        y_score,
        fpr_target=args.fpr_target,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "holdout_evaluation.json"

    payload = {
        "model_path": str(args.model),
        "features_csv": str(args.features_csv),
        "feature_columns": feature_cols,
        "fpr_target": args.fpr_target,
        "model_threshold": model_threshold,
        "model_fpr_target": model_fpr,
        "metrics": metrics,
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Saved holdout evaluation to %s", output_path)

    predictions_path = reports_dir / "holdout_predictions.pkl"
    joblib.dump({"y_true": y_true, "y_score": y_score}, predictions_path)
    logger.info("Saved holdout predictions to %s", predictions_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
