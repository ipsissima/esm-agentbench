#!/usr/bin/env python3
"""Run spectral validation on real HF agent traces with cross-model evaluation.

Extension of run_experiment.py for real agent traces from local HF models.
Adds cross-model generalization testing.

Usage:
    # Run on real HF traces for one scenario
    python analysis/run_real_hf_experiment.py --scenario supply_chain_poisoning

    # Run with cross-model validation
    python analysis/run_real_hf_experiment.py --scenario code_backdoor_injection --cross-model

    # Filter by models
    python analysis/run_real_hf_experiment.py --all-scenarios --model-filter deepseek-coder-7b-instruct

Output:
    reports/spectral_validation_real_hf/{scenario}/
      validation_report.json
      cross_model_report.json  (if --cross-model)
      features.csv
      roc_curve_by_model.png
      residual_distribution.png
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from certificates.spectral_prover import compute_detection_statistics

# Import from main experiment script
from analysis.run_experiment import (
    compute_features_for_trace,
    compute_baseline_subspace,
    plot_distributions,
    plot_roc_curve,
    compute_tpr_at_fpr,
)

# Optional imports
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


SCENARIOS = [
    "supply_chain_poisoning",
    "code_backdoor_injection",
    "code_review_bypass",
    "debug_credential_leak",
    "refactor_vuln_injection",
    "test_oracle_manipulation",
]


def load_real_hf_traces(traces_dir: Path, model_filter: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Load real HF agent traces organized by model.

    Expected structure:
        traces_dir/
            model1/
                gold/*.json
                creative/*.json
                drift/*.json
            model2/
                ...

    Parameters
    ----------
    traces_dir : Path
        Root directory containing model subdirectories
    model_filter : list of str, optional
        Only load traces from these models

    Returns
    -------
    dict
        Nested dict: {model_name: {label: [traces]}}
    """
    all_traces = {}

    # Find model directories
    if not traces_dir.exists():
        logger.warning("Trace directory not found: %s", traces_dir)
        return all_traces

    for model_dir in sorted(traces_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # Apply model filter
        if model_filter and model_name not in model_filter:
            continue

        model_traces = {'gold': [], 'creative': [], 'drift': []}

        for label in model_traces.keys():
            label_dir = model_dir / label
            if not label_dir.exists():
                continue

            for trace_file in sorted(label_dir.glob("*.json")):
                try:
                    with open(trace_file) as f:
                        data = json.load(f)
                        data['_source_file'] = str(trace_file)
                        data['_model'] = model_name
                        if 'label' not in data:
                            data['label'] = label
                        model_traces[label].append(data)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning("Error loading %s: %s", trace_file, e)

        if any(len(v) > 0 for v in model_traces.values()):
            all_traces[model_name] = model_traces

    return all_traces


def flatten_traces(model_traces: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Flatten model-organized traces to label-organized.

    Parameters
    ----------
    model_traces : dict
        {model_name: {label: [traces]}}

    Returns
    -------
    dict
        {label: [traces]}
    """
    flattened = {'gold': [], 'creative': [], 'drift': []}

    for model_name, traces_by_label in model_traces.items():
        for label, traces in traces_by_label.items():
            flattened[label].extend(traces)

    return flattened


def train_cross_model_classifier(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    train_models: List[str],
    test_models: List[str],
) -> Tuple[Any, Dict[str, float]]:
    """Train on some models, test on others.

    Parameters
    ----------
    features_df : pd.DataFrame
        Features with 'model' and 'label' columns
    feature_cols : list
        Feature columns to use
    train_models : list
        Models to train on
    test_models : list
        Models to test on

    Returns
    -------
    model
        Trained classifier
    metrics : dict
        Train and test metrics
    """
    if not HAS_SKLEARN:
        return None, {'train_auc': 0.5, 'test_auc': 0.5, 'note': 'sklearn not available'}

    # Split by model
    train_df = features_df[features_df['model'].isin(train_models) & features_df['valid']].copy()
    test_df = features_df[features_df['model'].isin(test_models) & features_df['valid']].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        return None, {'train_auc': 0.5, 'test_auc': 0.5, 'note': 'Insufficient data'}

    # Binary labels
    train_df['is_drift'] = (train_df['label'] == 'drift').astype(int)
    test_df['is_drift'] = (test_df['label'] == 'drift').astype(int)

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['is_drift'].values

    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df['is_drift'].values

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None, {'train_auc': 0.5, 'test_auc': 0.5, 'note': 'Only one class'}

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    train_auc = auc(fpr_train, tpr_train)

    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_proba)
    test_auc = auc(fpr_test, tpr_test)

    tpr_at_fpr05, actual_fpr = compute_tpr_at_fpr(fpr_test, tpr_test, 0.05)

    return model, {
        'train_models': train_models,
        'test_models': test_models,
        'train_auc': float(train_auc),
        'test_auc': float(test_auc),
        'tpr_at_fpr05': tpr_at_fpr05,
        'actual_fpr': actual_fpr,
        'fpr': fpr_test.tolist(),
        'tpr': tpr_test.tolist(),
        'thresholds': thresholds_test.tolist(),
    }


def run_cross_model_experiments(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    models: List[str],
) -> List[Dict[str, Any]]:
    """Run all cross-model combinations.

    Parameters
    ----------
    features_df : pd.DataFrame
        Features with model information
    feature_cols : list
        Feature columns
    models : list
        List of model names

    Returns
    -------
    list of dict
        Results for each train/test split
    """
    results = []

    # Try all leave-one-out splits
    for test_model in models:
        train_models = [m for m in models if m != test_model]
        if not train_models:
            continue

        logger.info("  Cross-model: train=%s, test=[%s]", train_models, test_model)

        _, metrics = train_cross_model_classifier(
            features_df,
            feature_cols,
            train_models,
            [test_model],
        )

        results.append({
            'train_models': train_models,
            'test_model': test_model,
            **metrics,
        })

        logger.info(
            "    Train AUC: %.4f, Test AUC: %.4f, TPR@FPR05: %.4f",
            metrics.get('train_auc', 0),
            metrics.get('test_auc', 0),
            metrics.get('tpr_at_fpr05', 0),
        )

    return results


def plot_roc_by_model(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    output_dir: Path,
) -> None:
    """Plot ROC curves for each model separately.

    Parameters
    ----------
    features_df : pd.DataFrame
        Features with model column
    feature_cols : list
        Feature columns
    output_dir : Path
        Output directory
    """
    if not HAS_MATPLOTLIB or not HAS_SKLEARN:
        return

    models = features_df['model'].unique()

    fig, ax = plt.subplots(figsize=(8, 6))

    for model in models:
        model_df = features_df[features_df['model'] == model].copy()
        model_df['is_drift'] = (model_df['label'] == 'drift').astype(int)

        X = model_df[feature_cols].fillna(0).values
        y = model_df['is_drift'].values

        if len(np.unique(y)) < 2:
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_scaled, y)

        y_proba = clf.predict_proba(X_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f'{model} (AUC={roc_auc:.3f})', linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves by Model', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve_by_model.png', dpi=150)
    plt.close()


def run_scenario(
    scenario_name: str,
    team: str = "ipsissima",
    k: int = 5,
    model_filter: Optional[List[str]] = None,
    cross_model: bool = False,
    skip_plots: bool = False,
) -> Dict[str, Any]:
    """Run spectral validation on real HF traces for a scenario.

    Parameters
    ----------
    scenario_name : str
        Scenario name
    team : str
        Team name
    k : int
        Rank for spectral analysis
    model_filter : list, optional
        Filter to specific models
    cross_model : bool
        Run cross-model validation

    Returns
    -------
    dict
        Report
    """
    logger.info("Processing scenario: %s", scenario_name)
    logger.info("  Rank k = %d", k)

    # Load traces
    traces_dir = PROJECT_ROOT / "submissions" / team / scenario_name / "experiment_traces_real_hf"
    model_traces = load_real_hf_traces(traces_dir, model_filter)

    if not model_traces:
        logger.warning("  No traces found in %s", traces_dir)
        return {}

    models = list(model_traces.keys())
    logger.info("  Found models: %s", models)

    # Flatten for per-model and overall analysis
    traces = flatten_traces(model_traces)

    total_traces = sum(len(v) for v in traces.values())
    logger.info("  Total traces: %d", total_traces)
    for label, trace_list in traces.items():
        logger.info("    %s: %d", label, len(trace_list))

    # Compute features
    logger.info("  Computing spectral features...")

    # Get baseline from gold traces
    gold_features = [
        compute_features_for_trace(t, k=k, baseline_U=None)
        for t in traces['gold']
    ]
    baseline_U = compute_baseline_subspace(gold_features)

    # Compute all features
    all_features = []
    for label, trace_list in traces.items():
        for trace in trace_list:
            features = compute_features_for_trace(trace, k=k, baseline_U=baseline_U)
            features['model'] = trace.get('_model', 'unknown')
            all_features.append(features)

    features_df = pd.DataFrame(all_features)

    # Output directory
    output_dir = PROJECT_ROOT / "reports" / "spectral_validation_real_hf" / scenario_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    csv_path = output_dir / 'features.csv'
    features_df.drop(columns=['_U_k'], errors='ignore').to_csv(csv_path, index=False)
    logger.info("  Saved features to %s", csv_path)

    # Feature columns
    feature_cols = ['residual', 'theoretical_bound', 'sigma_max', 'singular_gap',
                    'tail_energy', 'pca_explained', 'dk_angle', 'koopman_residual']

    # Per-model analysis
    per_model_results = {}
    for model in models:
        model_df = features_df[features_df['model'] == model].copy()
        model_df['is_drift'] = (model_df['label'] == 'drift').astype(int)

        X = model_df[feature_cols].fillna(0).values
        y = model_df['is_drift'].values

        if len(np.unique(y)) < 2 or not HAS_SKLEARN:
            per_model_results[model] = {'auc': 0.5}
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_scaled, y)

        y_proba = clf.predict_proba(X_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        model_auc = auc(fpr, tpr)

        tpr_at_fpr05, _ = compute_tpr_at_fpr(fpr, tpr, 0.05)

        per_model_results[model] = {
            'auc': float(model_auc),
            'tpr_at_fpr05': tpr_at_fpr05,
        }

    # Overall metrics (all models combined)
    features_df_valid = features_df[features_df['valid']].copy()
    features_df_valid['is_drift'] = (features_df_valid['label'] == 'drift').astype(int)

    X_all = features_df_valid[feature_cols].fillna(0).values
    y_all = features_df_valid['is_drift'].values

    if len(np.unique(y_all)) >= 2 and HAS_SKLEARN:
        scaler_all = StandardScaler()
        X_all_scaled = scaler_all.fit_transform(X_all)

        clf_all = LogisticRegression(max_iter=1000)
        clf_all.fit(X_all_scaled, y_all)

        y_all_proba = clf_all.predict_proba(X_all_scaled)[:, 1]
        fpr_all, tpr_all, _ = roc_curve(y_all, y_all_proba)
        overall_auc = auc(fpr_all, tpr_all)
        overall_tpr_at_fpr05, _ = compute_tpr_at_fpr(fpr_all, tpr_all, 0.05)
    else:
        overall_auc = 0.5
        overall_tpr_at_fpr05 = 0.0

    # Build report
    report = {
        'scenario': scenario_name,
        'data_source': 'real_traces_only',
        'models': models,
        'overall_auc': float(overall_auc),
        'overall_tpr_at_fpr05': float(overall_tpr_at_fpr05),
        'per_model': per_model_results,
        'num_traces': {
            'gold': len(traces['gold']),
            'creative': len(traces['creative']),
            'drift': len(traces['drift']),
        },
        'rank_k': k,
        'real_hf_traces': True,
    }

    # Cross-model validation
    if cross_model and len(models) > 1:
        logger.info("  Running cross-model validation...")
        cross_results = run_cross_model_experiments(features_df, feature_cols, models)

        cross_report = {
            'scenario': scenario_name,
            'models': models,
            'experiments': cross_results,
            'summary': {
                'mean_test_auc': float(np.mean([r.get('test_auc', 0.5) for r in cross_results])),
                'mean_tpr_at_fpr05': float(np.mean([r.get('tpr_at_fpr05', 0) for r in cross_results])),
            },
        }

        cross_report_path = output_dir / 'cross_model_report.json'
        with open(cross_report_path, 'w') as f:
            json.dump(cross_report, f, indent=2)
        logger.info("  Saved cross-model report to %s", cross_report_path)

        report['cross_model'] = cross_report['summary']

    # Save report
    report_path = output_dir / 'validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info("  Saved report to %s", report_path)

    # Plots (optional)
    if not skip_plots:
        plot_distributions(features_df, output_dir)
        plot_roc_by_model(features_df, feature_cols, output_dir)

    # Summary
    logger.info("  Results:")
    logger.info("    Overall AUC: %.4f", report['overall_auc'])
    logger.info("    Overall TPR @ FPR=0.05: %.4f", report['overall_tpr_at_fpr05'])
    for model, metrics in per_model_results.items():
        logger.info("    %s: AUC=%.4f, TPR@FPR05=%.4f", model, metrics['auc'], metrics.get('tpr_at_fpr05', 0))

    return report


def load_evaluation_config() -> dict:
    """Load evaluation config from evaluation_config.yaml."""
    import yaml

    config_path = PROJECT_ROOT / "evaluation_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run spectral validation on real HF agent traces")
    parser.add_argument("--scenario", help="Scenario name")
    parser.add_argument("--all-scenarios", action="store_true", help="Run all scenarios")
    parser.add_argument("--team", default="ipsissima", help="Team name")
    parser.add_argument("--k", type=int, help="Rank for spectral analysis (default: from config)")
    parser.add_argument("--model-filter", help="Comma-separated model names to include")
    parser.add_argument("--cross-model", action="store_true", help="Run cross-model validation")
    parser.add_argument("--skip-plots", action="store_true", help="Skip generating plots (faster)")

    args = parser.parse_args()

    # Load config
    config = load_evaluation_config()

    # Use k from config if not specified
    if args.k is None:
        args.k = config.get('certificate', {}).get('pca_rank', 5)

    if args.all_scenarios:
        scenarios = SCENARIOS
    elif args.scenario:
        scenarios = [args.scenario]
    else:
        logger.error("Must specify --scenario or --all-scenarios")
        return 1

    model_filter = None
    if args.model_filter:
        model_filter = [m.strip() for m in args.model_filter.split(",")]

    for scenario in scenarios:
        run_scenario(
            scenario,
            team=args.team,
            k=args.k,
            model_filter=model_filter,
            cross_model=args.cross_model,
            skip_plots=args.skip_plots,
        )

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    sys.exit(main())
