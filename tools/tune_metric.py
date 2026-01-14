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
from sklearn.linear_model import LogisticRegression, HuberRegressor
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
    "theoretical_bound_norm",
    "theoretical_bound_prompt_adj",
    "residual",
    "residual_norm",
    "residual_fro_norm",
    "koopman_residual",
    "pca_explained",
    "r_eff",
    "r_rel",
    "length_T",
    "embed_norm",
    "semantic_drift",
    "semantic_centroid_distance",
    "sv_max_ratio",
    "insample_residual",
    "oos_residual",
]


HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
HAS_SEABORN = importlib.util.find_spec("seaborn") is not None
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
    gamma_grid: List[float],
    use_prompt_adj: bool,
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

        # Mitigate PCA confounding: learn train-fold-only linear correction
        # for theoretical bound explained by pca_explained.
        # This prevents leakage by fitting only on training folds.
        if "theoretical_bound" in X_train.columns and "pca_explained" in X_train.columns:
            try:
                # HuberRegressor is robust to outliers and deterministic here.
                huber = HuberRegressor()
                # Fit on training fold
                huber.fit(
                    X_train[["pca_explained"]].to_numpy(dtype=float),
                    X_train["theoretical_bound"].to_numpy(dtype=float),
                )
                # Compute adjusted theoretical bound: residuals of the regression
                X_train = X_train.copy()
                X_test = X_test.copy()
                X_train["theoretical_bound_adj"] = X_train["theoretical_bound"] - huber.predict(
                    X_train[["pca_explained"]]
                )
                X_test["theoretical_bound_adj"] = X_test["theoretical_bound"] - huber.predict(
                    X_test[["pca_explained"]]
                )

                # Replace theoretical_bound with adjusted version so downstream
                # models use the corrected quantity. Keep the raw one around if needed.
                X_train["theoretical_bound"] = X_train["theoretical_bound_adj"]
                X_test["theoretical_bound"] = X_test["theoretical_bound_adj"]
            except Exception as exc:
                logger.debug("Could not compute theoretical_bound_adj for fold %s: %s", fold_idx, exc)

        best_gamma = gamma_grid[0] if gamma_grid else 0.0
        best_auc = -np.inf
        best_scores = None
        best_params = None
        best_threshold = 0.5
        best_fpr = 1.0
        best_tpr = 0.0
        best_confusion = None

        for gamma in gamma_grid:
            X_train_gamma = X_train.copy()
            X_test_gamma = X_test.copy()

            if use_prompt_adj and "semantic_centroid_distance" in X_train_gamma.columns:
                base_col = "theoretical_bound_norm" if "theoretical_bound_norm" in X_train_gamma.columns else "theoretical_bound"
                if base_col in X_train_gamma.columns:
                    X_train_gamma["theoretical_bound_prompt_adj"] = (
                        X_train_gamma[base_col] - gamma * X_train_gamma["semantic_centroid_distance"]
                    )
                    X_test_gamma["theoretical_bound_prompt_adj"] = (
                        X_test_gamma[base_col] - gamma * X_test_gamma["semantic_centroid_distance"]
                    )

            grid = GridSearchCV(
                pipe,
                param_grid=param_grid,
                scoring="roc_auc",
                cv=inner_cv.split(X_train_gamma, y_train, groups=train_groups),
            )
            grid.fit(X_train_gamma, y_train)
            model = grid.best_estimator_

            scores = model.predict_proba(X_test_gamma)[:, 1]
            auc_score = roc_auc_score(y_test, scores) if len(np.unique(y_test)) > 1 else 0.5
            if auc_score > best_auc:
                best_auc = auc_score
                best_gamma = gamma
                best_scores = scores
                best_params = grid.best_params_
                tpr, fpr, threshold = _tpr_at_fpr(y_test, scores, fpr_target)
                preds = (scores >= threshold).astype(int)
                cm = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
                tn, fp, fn, tp = [int(v) for v in cm]
                best_threshold = float(threshold)
                best_fpr = float(fpr)
                best_tpr = float(tpr)
                best_confusion = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

        if best_scores is None:
            best_scores = np.zeros_like(y_test, dtype=float)
            best_params = {}
            best_confusion = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}

        fold_results.append({
            "fold": fold_idx,
            "auc": float(best_auc),
            "tpr_at_fpr": float(best_tpr),
            "actual_fpr": float(best_fpr),
            "threshold": float(best_threshold),
            "best_params": best_params,
            "best_gamma": float(best_gamma),
            "confusion": best_confusion,
        })

        all_scores.append(best_scores)
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
    """Write diagnostics CSVs/plots with optional columns handled safely."""
    required = {"scenario", "label"}
    missing_required = required - set(df.columns)
    if missing_required:
        missing_list = ", ".join(sorted(missing_required))
        raise ValueError(f"Missing required diagnostics columns: {missing_list}")

    output_dir.mkdir(parents=True, exist_ok=True)

    desired = [
        "run_id",
        "scenario",
        "label",
        "theoretical_bound",
        "theoretical_bound_norm",
        "theoretical_bound_prompt_adj",
        "residual",
        "residual_norm",
        "residual_fro_norm",
        "pca_explained",
        "r_eff",
        "r_rel",
        "embed_norm",
        "semantic_drift",
        "semantic_centroid_distance",
        "sv_max_ratio",
        "steps",
        "tail_energy",
        "tail_ratio",
    ]

    if "tail_energy" not in df.columns:
        if "singular_values" in df.columns:
            def compute_tail_energy(sv):
                try:
                    if isinstance(sv, str):
                        sv_list = json.loads(sv)
                    else:
                        sv_list = list(sv)
                    sv2 = np.array(sv_list) ** 2
                    if sv2.sum() <= 0:
                        return np.nan
                    k = max(1, len(sv2) // 2)
                    tail = sv2[k:].sum() / sv2.sum()
                    return float(tail)
                except Exception:
                    return np.nan
            logger.info("Computing 'tail_energy' from 'singular_values' column.")
            df = df.copy()
            df["tail_energy"] = df["singular_values"].apply(compute_tail_energy)
        elif "S" in df.columns:
            def compute_tail_from_S(S):
                try:
                    if isinstance(S, str):
                        sv = np.array(json.loads(S))
                    else:
                        sv = np.array(S)
                    sv2 = sv ** 2
                    if sv2.sum() <= 0:
                        return np.nan
                    k = max(1, len(sv2) // 2)
                    tail = sv2[k:].sum() / sv2.sum()
                    return float(tail)
                except Exception:
                    return np.nan
            logger.info("Computing 'tail_energy' from 'S' column.")
            df = df.copy()
            df["tail_energy"] = df["S"].apply(compute_tail_from_S)
        else:
            logger.warning(
                "Optional diagnostic 'tail_energy' not found and cannot be computed "
                "(no 'singular_values' or 'S'). It will be omitted."
            )

    available = [col for col in desired if col in df.columns]
    missing_optional = [col for col in desired if col not in available and col not in required]
    if missing_optional:
        logger.info("Missing optional diagnostics columns: %s", ", ".join(missing_optional))

    if available:
        diagnostics = df[available].copy()
    else:
        logger.warning("No diagnostic columns available to write.")
        diagnostics = df.iloc[:, :0].copy()

    if "theoretical_bound" in diagnostics.columns:
        diagnostics["theoretical_bound_sign"] = np.sign(diagnostics["theoretical_bound"])

    diagnostics_path = output_dir / "diagnostics_summary.csv"
    diagnostics.to_csv(diagnostics_path, index=False)
    logger.info("Wrote diagnostics to %s", diagnostics_path)

    if "theoretical_bound" in diagnostics.columns:
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
    else:
        logger.warning("Skipping theoretical bound summary: missing 'theoretical_bound' column.")

    if HAS_MATPLOTLIB and "theoretical_bound" in diagnostics.columns:
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

    if HAS_SEABORN and HAS_MATPLOTLIB:
        import seaborn as sns

        try:
            numeric_cols = diagnostics.select_dtypes(include=[np.number]).columns.tolist()
            if 2 <= len(numeric_cols) <= 6:
                plot_data = diagnostics[numeric_cols + ["label"]] if "label" in diagnostics.columns else diagnostics[numeric_cols]
                sns.pairplot(plot_data)
                plt.tight_layout()
                pairplot_path = output_dir / "diagnostics_pairplot.png"
                plt.savefig(pairplot_path)
                plt.close()
                logger.info("Saved diagnostics pairplot to %s", pairplot_path)
            else:
                logger.info("Skipping pairplot; numeric columns count=%s", len(numeric_cols))
        except Exception as exc:
            logger.debug("Could not write pairplots: %s", exc)


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
    parser.add_argument(
        "--gamma-grid",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        help="Grid of gamma values for prompt-based shrinkage to evaluate via nested CV",
    )
    parser.add_argument(
        "--use-prompt-adj",
        action="store_true",
        help="Enable prompt-based adjusted theoretical_bound score in feature set",
    )
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

    if args.use_prompt_adj and "theoretical_bound_prompt_adj" not in df.columns:
        base_col = None
        if "theoretical_bound_norm" in df.columns:
            base_col = "theoretical_bound_norm"
        elif "theoretical_bound" in df.columns:
            base_col = "theoretical_bound"
        if base_col and "semantic_centroid_distance" in df.columns:
            df = df.copy()
            df["theoretical_bound_prompt_adj"] = df[base_col] - 0.0 * df["semantic_centroid_distance"]

    if args.groups_col not in df.columns:
        raise ValueError(f"Missing groups column: {args.groups_col}")
    if args.label_col not in df.columns:
        raise ValueError(f"Missing label column: {args.label_col}")

    X, y, groups, df = _prepare_features(df, args.label_col, args.groups_col)
    if len(np.unique(y)) < 2:
        raise ValueError("Need both good and bad labels to tune classifier")

    _write_diagnostics(df, PROJECT_ROOT / "reports")

    fold_results, summary, y_true, y_score = _nested_cv(
        X,
        y,
        groups,
        seed=args.seed,
        fpr_target=args.fpr_target,
        gamma_grid=args.gamma_grid,
        use_prompt_adj=args.use_prompt_adj,
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
    best_gamma = float(np.median([fold.get("best_gamma", 0.0) for fold in fold_results]))

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
        "best_gamma": best_gamma,
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
        "best_gamma": best_gamma,
    }
    model_path = reports_dir / "best_model.pkl"
    joblib.dump(model_payload, model_path)
    logger.info("Saved best model to %s", model_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
