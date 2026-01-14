#!/usr/bin/env python3
"""Tune augmented drift detection classifier with nested CV."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import importlib.util
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tools.feature_utils import (
    NormalizationConfig,
    PROJECT_ROOT,
    discover_trace_dirs,
    infer_scenario_from_path,
    iter_trace_files,
    label_to_binary,
    load_trace_json,
    compute_trace_features,
)

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "theoretical_bound",
    "residual",
    "koopman_residual",
    "pca_explained",
    "r_eff",
    "length_T",
    "embed_norm",
    "semantic_drift",
    "insample_residual",
    "oos_residual",
]


HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
if HAS_MATPLOTLIB:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float) -> Tuple[float, float, float]:
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0, 1.0
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.where(fpr <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0, float(fpr[0]) if len(fpr) else 1.0, float(thresholds[0]) if len(thresholds) else 0.5
    best = idx[-1]
    return float(tpr[best]), float(fpr[best]), float(thresholds[best])


def _bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(values), size=len(values))
        samples.append(float(np.mean(values[idx])))
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def _bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
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


def _prepare_features(df: pd.DataFrame, label_col: str, groups_col: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    df = df.copy()
    df["label_binary"] = df[label_col].apply(label_to_binary)
    df = df.dropna(subset=["label_binary"]).copy()
    y = df["label_binary"].astype(int).to_numpy()
    groups = df[groups_col].astype(str).to_numpy()

    feature_cols = [col for col in FEATURE_COLUMNS if col in df.columns]
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        logger.warning("Missing feature columns: %s", missing)
    X = df[feature_cols].fillna(0.0)
    return X, y, groups, df


def _nested_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    fpr_target: float,
) -> Tuple[List[Dict[str, float]], Dict[str, float], np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    outer_splits = min(5, len(unique_groups))
    if outer_splits < 2:
        raise ValueError("Need at least 2 groups for outer GroupKFold")
    inner_splits = min(3, max(2, len(unique_groups) - 1))

    outer_cv = GroupKFold(n_splits=outer_splits)
    inner_cv = GroupKFold(n_splits=inner_splits)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)),
    ])
    param_grid = {"clf__C": [1e-3, 1e-2, 1e-1, 1, 10, 100]}

    fold_results: List[Dict[str, float]] = []
    all_scores = []
    all_labels = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_test, y_test = X.iloc[test_idx], y[test_idx]
        train_groups = groups[train_idx]

        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=inner_cv.split(X_train, y_train, groups=train_groups),
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        scores = best_model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, scores) if len(np.unique(y_test)) > 1 else 0.5
        tpr, fpr, threshold = _tpr_at_fpr(y_test, scores, fpr_target)
        preds = (scores >= threshold).astype(int)
        cm = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        tn, fp, fn, tp = [int(v) for v in cm]

        fold_results.append({
            "fold": fold_idx,
            "auc": float(auc_score),
            "tpr_at_fpr": float(tpr),
            "actual_fpr": float(fpr),
            "threshold": float(threshold),
            "best_params": grid.best_params_,
            "confusion": {
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            },
        })

        all_scores.append(scores)
        all_labels.append(y_test)

    all_scores_arr = np.concatenate(all_scores)
    all_labels_arr = np.concatenate(all_labels)

    summary = {
        "auc_mean": float(np.mean([f["auc"] for f in fold_results])),
        "auc_se": float(np.std([f["auc"] for f in fold_results], ddof=1) / np.sqrt(len(fold_results))),
        "tpr_mean": float(np.mean([f["tpr_at_fpr"] for f in fold_results])),
        "tpr_se": float(np.std([f["tpr_at_fpr"] for f in fold_results], ddof=1) / np.sqrt(len(fold_results))),
    }

    return fold_results, summary, all_labels_arr, all_scores_arr


def _write_diagnostics(df: pd.DataFrame, output_dir: Path) -> None:
    diagnostics = df[[
        "run_id",
        "scenario",
        "label",
        "theoretical_bound",
        "residual",
        "tail_energy",
        "pca_explained",
    ]].copy()
    diagnostics["theoretical_bound_sign"] = np.sign(diagnostics["theoretical_bound"])  # -1, 0, 1

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = output_dir / "theoretical_bound_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False)
    logger.info("Wrote theoretical bound diagnostics to %s", diagnostics_path)

    summary = diagnostics.groupby(["scenario", "label"]).agg(
        count=("theoretical_bound", "count"),
        mean=("theoretical_bound", "mean"),
        min=("theoretical_bound", "min"),
        max=("theoretical_bound", "max"),
        negative_rate=("theoretical_bound", lambda x: float(np.mean(x < 0))),
    )
    summary_path = output_dir / "theoretical_bound_summary.csv"
    summary.reset_index().to_csv(summary_path, index=False)
    logger.info("Wrote theoretical bound summary to %s", summary_path)

    if HAS_MATPLOTLIB:
        plt.figure(figsize=(8, 5))
        for label, group in diagnostics.groupby("label"):
            plt.hist(group["theoretical_bound"].values, bins=30, alpha=0.5, label=label)
        plt.legend()
        plt.xlabel("Theoretical bound")
        plt.ylabel("Count")
        plt.title("Theoretical Bound Distribution by Label")
        plt.tight_layout()
        plot_path = output_dir / "theoretical_bound_hist.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info("Saved theoretical bound histogram to %s", plot_path)


def _recompute_features(
    features_csv: Path,
    normalization: NormalizationConfig,
    seed: int,
) -> pd.DataFrame:
    trace_dirs = discover_trace_dirs([
        PROJECT_ROOT / "experiment_traces",
        PROJECT_ROOT / "submissions",
        PROJECT_ROOT / "scenarios",
    ])

    if not trace_dirs:
        raise FileNotFoundError("No experiment_traces directories found to recompute features")

    rows = []
    for trace_dir in trace_dirs:
        scenario = infer_scenario_from_path(trace_dir)
        for trace_file, label in iter_trace_files(trace_dir):
            trace_data = load_trace_json(trace_file)
            embeddings = trace_data.get("embeddings")
            if embeddings is None:
                logger.warning("Skipping %s: no embeddings", trace_file)
                continue
            features = compute_trace_features(
                np.asarray(embeddings, dtype=np.float64),
                normalization=normalization,
                k=10,
                kernel_strict=False,
            )
            rows.append({
                "run_id": trace_data.get("run_id", trace_file.stem),
                "scenario": trace_data.get("scenario", scenario),
                "label": trace_data.get("label", label),
                **features,
            })

    df = pd.DataFrame(rows)
    features_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_csv, index=False)
    logger.info("Saved recomputed features to %s", features_csv)
    return df


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Tune augmented drift detection metric.")
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--groups-col", type=str, default="scenario")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--fpr-target", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--l2-normalize-steps", action="store_true")
    parser.add_argument("--zscore-per-trace", action="store_true")
    parser.add_argument("--length-normalize", action="store_true")
    parser.add_argument("--trim-proportion", type=float, default=0.0)

    args = parser.parse_args()

    normalization = NormalizationConfig(
        l2_normalize_steps=args.l2_normalize_steps,
        zscore_per_trace=args.zscore_per_trace,
        length_normalize=args.length_normalize,
        trim_proportion=args.trim_proportion,
    )

    if args.recompute or not args.features_csv.exists():
        df = _recompute_features(args.features_csv, normalization, seed=args.seed)
    else:
        df = pd.read_csv(args.features_csv)

    if args.groups_col not in df.columns:
        raise ValueError(f"Missing groups column: {args.groups_col}")
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column: {args.label_col}")

    X, y, groups, df = _prepare_features(df, args.label_col, args.groups_col)
    if len(np.unique(y)) < 2:
        raise ValueError("Need both good and bad labels to tune classifier")

    _write_diagnostics(df, PROJECT_ROOT / "reports")

    fold_results, summary, y_true, y_score = _nested_cv(
        X, y, groups, seed=args.seed, fpr_target=args.fpr_target
    )

    auc_ci = _bootstrap_auc(y_true, y_score, n_boot=args.n_boot, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    tpr_vals = []
    for _ in range(args.n_boot):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        tpr_val, _, _ = _tpr_at_fpr(y_true[idx], y_score[idx], args.fpr_target)
        tpr_vals.append(tpr_val)
    tpr_ci = (float(np.percentile(tpr_vals, 2.5)), float(np.percentile(tpr_vals, 97.5)))

    # Train final model on full dataset with best params from mean fold
    best_params = sorted(
        fold_results,
        key=lambda r: r["auc"],
        reverse=True,
    )[0]["best_params"]

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=args.seed)),
    ])
    final_model.set_params(**best_params)
    final_model.fit(X, y)
    final_scores = final_model.predict_proba(X)[:, 1]
    tpr_full, fpr_full, threshold_full = _tpr_at_fpr(y, final_scores, args.fpr_target)

    results = {
        "features_csv": str(args.features_csv),
        "feature_columns": X.columns.tolist(),
        "label_mapping": {
            "good": ["coherent", "creative", "gold", "good"],
            "bad": ["drift", "poison", "starvation", "bad"],
        },
        "outer_folds": fold_results,
        "summary": summary,
        "auc_ci": {"low": auc_ci[0], "high": auc_ci[1]},
        "tpr_ci": {"low": tpr_ci[0], "high": tpr_ci[1]},
        "fpr_target": args.fpr_target,
        "selected_threshold": threshold_full,
        "selected_tpr": tpr_full,
        "selected_fpr": fpr_full,
    }

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_path = reports_dir / "tuning_results.json"
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    logger.info("Saved tuning results to %s", results_path)

    model_payload = {
        "model": final_model,
        "feature_columns": X.columns.tolist(),
        "threshold": threshold_full,
        "fpr_target": args.fpr_target,
    }
    model_path = reports_dir / "best_model.pkl"
    joblib.dump(model_payload, model_path)
    logger.info("Saved best model to %s", model_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
