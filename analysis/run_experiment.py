#!/usr/bin/env python3
"""Run spectral validation experiments on Phase-1 scenarios.

This script:
1. Loads labelled traces (gold/creative/drift) from experiment_traces/
2. Computes spectral certificates for each trace
3. Trains a classifier to distinguish drift from creative
4. Computes ROC/AUC metrics and saves validation reports

For real agent evaluation, generate traces using:
    tools/real_agents_hf/run_real_agents.py

Usage:
    python analysis/run_experiment.py --all-scenarios
    python analysis/run_experiment.py --scenario code_backdoor_injection --k 10

Output:
    reports/spectral_validation/{scenario}/
      validation_report.json
      features.csv
      roc_curve.png
      residual_distribution.png
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from certificates.spectral_prover import (
    compute_detection_statistics,
    compute_theoretical_bound,
)

# Optional imports for visualization and ML
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available; plots will be skipped")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc, classification_report
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not available; classification will use heuristics")


def load_traces(traces_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load all traces from experiment_traces directory.

    Parameters
    ----------
    traces_dir : Path
        Root directory containing gold/, creative/, drift/ subdirectories.

    Returns
    -------
    dict
        Mapping from label to list of trace data dicts.
    """
    traces: Dict[str, List[Dict[str, Any]]] = {'gold': [], 'creative': [], 'drift': []}

    for label in traces:
        label_dir = traces_dir / label
        if not label_dir.exists():
            continue

        for trace_file in sorted(label_dir.glob("*.json")):
            try:
                with open(trace_file, 'r') as f:
                    data = json.load(f)
                    data['_source_file'] = str(trace_file)
                    traces[label].append(data)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Error loading %s: %s", trace_file, e)

    return traces


def compute_features_for_trace(
    trace_data: Dict[str, Any],
    k: int,
    baseline_U: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute all spectral features for a single trace.

    Parameters
    ----------
    trace_data : dict
        Trace data with 'embeddings' and 'label' keys.
    k : int
        Rank for spectral analysis.
    baseline_U : np.ndarray, optional
        Baseline subspace for Davis-Kahan angle.

    Returns
    -------
    dict
        Feature dict with run_id, label, and all spectral metrics.
    """
    embeddings = trace_data.get('embeddings', [])
    if not embeddings or len(embeddings) < 2:
        return {
            'run_id': trace_data.get('run_id', 'unknown'),
            'label': trace_data.get('label', 'unknown'),
            'residual': 1.0,
            'theoretical_bound': float('inf'),
            'sigma_max': 0.0,
            'singular_gap': 0.0,
            'tail_energy': 1.0,
            'pca_explained': 0.0,
            'dk_angle': 0.0,
            'koopman_residual': 1.0,
            'length_T': len(embeddings),
            'valid': False,
        }

    # Compute detection statistics
    stats = compute_detection_statistics(
        run_trace=embeddings,
        k=k,
        baseline_U=baseline_U,
    )

    return {
        'run_id': trace_data.get('run_id', 'unknown'),
        'label': trace_data.get('label', 'unknown'),
        'residual': stats['residual'],
        'theoretical_bound': stats['theoretical_bound'],
        'sigma_max': stats['sigma_max'],
        'singular_gap': stats['singular_gap'],
        'tail_energy': stats['tail_energy'],
        'pca_explained': stats['pca_explained'],
        'dk_angle': stats['dk_angle'],
        'koopman_residual': stats['koopman_residual'],
        'length_T': stats['length_T'],
        'valid': True,
        '_U_k': stats['U_k'],  # For baseline computation
    }


def compute_baseline_subspace(gold_features: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    """Compute baseline subspace from gold traces.

    Parameters
    ----------
    gold_features : list
        List of feature dicts from gold traces.

    Returns
    -------
    np.ndarray or None
        Average left singular vectors from gold traces.
    """
    valid_Uk = [f['_U_k'] for f in gold_features if f.get('valid') and '_U_k' in f]
    if not valid_Uk:
        return None

    # Use first valid U_k as baseline (could also average or use PCA)
    return valid_Uk[0]


def train_classifier(
    features_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[Any, Dict[str, float]]:
    """Train logistic regression classifier for drift detection.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with features and 'label' column.
    feature_cols : list
        Column names to use as features.

    Returns
    -------
    model
        Trained classifier.
    metrics : dict
        Cross-validation metrics.
    """
    # Binary classification: drift vs non-drift (creative + gold)
    df = features_df[features_df['valid']].copy()
    df['is_drift'] = (df['label'] == 'drift').astype(int)

    X = df[feature_cols].fillna(0).values
    y = df['is_drift'].values

    if len(np.unique(y)) < 2:
        return None, {'auc': 0.5, 'accuracy': 0.0, 'note': 'Only one class present'}

    if not HAS_SKLEARN:
        # Fallback: use residual threshold
        return None, {'auc': 0.5, 'accuracy': 0.0, 'note': 'sklearn not available'}

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train logistic regression with cross-validation
    model = LogisticRegression(max_iter=1000, random_state=42)

    # Cross-validation
    cv = StratifiedKFold(n_splits=min(5, len(y) // 2), shuffle=True, random_state=42)
    try:
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
    except ValueError:
        cv_scores = [0.5]

    # Fit final model
    model.fit(X_scaled, y)

    # Compute ROC curve on training data (for reporting)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    return model, {
        'auc': float(roc_auc),
        'cv_auc_mean': float(np.mean(cv_scores)),
        'cv_auc_std': float(np.std(cv_scores)),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
    }


def compute_tpr_at_fpr(
    fpr: np.ndarray,
    tpr: np.ndarray,
    target_fpr: float = 0.05,
) -> Tuple[float, float]:
    """Compute TPR at specified FPR threshold.

    Parameters
    ----------
    fpr, tpr : array
        ROC curve arrays.
    target_fpr : float
        Target false positive rate.

    Returns
    -------
    tpr_at_fpr : float
        True positive rate at or below target FPR.
    actual_fpr : float
        Actual FPR achieved.
    """
    idx = np.where(np.array(fpr) <= target_fpr)[0]
    if len(idx) == 0:
        return 0.0, fpr[0] if len(fpr) > 0 else 1.0

    best_idx = idx[-1]  # Last index where FPR <= target
    return float(tpr[best_idx]), float(fpr[best_idx])


def plot_distributions(
    features_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot residual and dk_angle distributions by label.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with features.
    output_dir : Path
        Output directory for plots.
    """
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residual distribution
    for label in ['gold', 'creative', 'drift']:
        data = features_df[features_df['label'] == label]['residual'].dropna()
        if len(data) > 0:
            axes[0].hist(data, bins=20, alpha=0.5, label=label)
    axes[0].set_xlabel('Residual')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Residual Distribution by Label')
    axes[0].legend()

    # DK angle distribution
    for label in ['gold', 'creative', 'drift']:
        data = features_df[features_df['label'] == label]['dk_angle'].dropna()
        if len(data) > 0:
            axes[1].hist(data, bins=20, alpha=0.5, label=label)
    axes[1].set_xlabel('Davis-Kahan Angle (radians)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('DK Angle Distribution by Label')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'residual_distribution.png', dpi=150)
    plt.close()


def plot_roc_curve(
    fpr: List[float],
    tpr: List[float],
    auc_score: float,
    output_dir: Path,
) -> None:
    """Plot ROC curve.

    Parameters
    ----------
    fpr, tpr : list
        ROC curve points.
    auc_score : float
        Area under curve.
    output_dir : Path
        Output directory.
    """
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.axvline(x=0.05, color='r', linestyle=':', label='FPR = 0.05')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve: Drift vs Creative Detection')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=150)
    plt.close()


def run_experiment(
    traces_dir: Path,
    output_dir: Path,
    k: int = 10,
    scenario_name: str = 'default',
) -> Dict[str, Any]:
    """Run full spectral validation experiment.

    Parameters
    ----------
    traces_dir : Path
        Directory with gold/, creative/, drift/ subdirectories.
    output_dir : Path
        Directory for output reports.
    k : int
        Rank for spectral analysis.
    scenario_name : str
        Name of scenario for reporting.

    Returns
    -------
    dict
        Validation report with AUC, TPR_at_FPR05, etc.
    """
    logger.info("Running experiment for scenario: %s", scenario_name)
    logger.info("  Traces dir: %s", traces_dir)
    logger.info("  Output dir: %s", output_dir)
    logger.info("  Rank k: %d", k)

    # Load traces
    traces = load_traces(traces_dir)
    logger.info(
        "  Loaded traces: gold=%d, creative=%d, drift=%d",
        len(traces['gold']), len(traces['creative']), len(traces['drift'])
    )

    if not any(traces.values()):
        return {
            'scenario': scenario_name,
            'error': 'No traces found',
            'AUC': 0.0,
            'TPR_at_FPR05': 0.0,
        }

    # First pass: compute features for gold traces to get baseline
    gold_features = [
        compute_features_for_trace(t, k=k, baseline_U=None)
        for t in traces['gold']
    ]

    # Get baseline subspace from gold traces
    baseline_U = compute_baseline_subspace(gold_features)

    # Second pass: compute all features with baseline
    all_features = []
    for label, trace_list in traces.items():
        for trace in trace_list:
            features = compute_features_for_trace(trace, k=k, baseline_U=baseline_U)
            all_features.append(features)

    # Create DataFrame
    features_df = pd.DataFrame(all_features)

    # Save features CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'features.csv'
    features_df.drop(columns=['_U_k'], errors='ignore').to_csv(csv_path, index=False)
    logger.info("  Saved features to %s", csv_path)

    # Train classifier and compute metrics
    feature_cols = ['residual', 'theoretical_bound', 'sigma_max', 'singular_gap',
                    'tail_energy', 'pca_explained', 'dk_angle', 'koopman_residual']

    model, metrics = train_classifier(features_df, feature_cols)

    # Compute TPR at FPR=0.05
    if 'fpr' in metrics and 'tpr' in metrics:
        tpr_at_fpr05, actual_fpr = compute_tpr_at_fpr(
            np.array(metrics['fpr']),
            np.array(metrics['tpr']),
            target_fpr=0.05,
        )
    else:
        tpr_at_fpr05, actual_fpr = 0.0, 1.0

    # Compute median residuals per label
    median_residuals = {}
    for label in ['gold', 'creative', 'drift']:
        data = features_df[features_df['label'] == label]['residual'].dropna()
        median_residuals[label] = float(data.median()) if len(data) > 0 else None

    # Compute calibrated threshold (FPR <= 0.05)
    if 'thresholds' in metrics and 'fpr' in metrics:
        fpr_arr = np.array(metrics['fpr'])
        idx = np.where(fpr_arr <= 0.05)[0]
        if len(idx) > 0:
            threshold_tau = float(metrics['thresholds'][idx[-1]])
        else:
            threshold_tau = float(metrics['thresholds'][0]) if metrics['thresholds'] else 0.5
    else:
        threshold_tau = median_residuals.get('gold', 0.0) + 0.3 if median_residuals.get('gold') else 0.5

    # Detect data source from traces
    data_sources = set()
    for label in ['gold', 'creative', 'drift']:
        for trace in traces[label]:
            ds = trace.get('data_source', 'unknown')
            data_sources.add(ds)
    
    # Determine overall data source
    if 'synthetic' in data_sources:
        data_source = 'synthetic'
    elif 'real_traces_only' in data_sources or all(ds == 'unknown' for ds in data_sources):
        # If all are unknown (legacy traces), assume real for backward compatibility
        # Otherwise, if any are explicitly real_traces_only, mark as real
        data_source = 'real_traces_only' if 'real_traces_only' in data_sources else 'unknown'
    else:
        data_source = 'unknown'

    # Build report
    report = {
        'scenario': scenario_name,
        'data_source': data_source,
        'AUC': metrics.get('auc', 0.5),
        'cv_AUC_mean': metrics.get('cv_auc_mean', 0.5),
        'cv_AUC_std': metrics.get('cv_auc_std', 0.0),
        'TPR_at_FPR05': tpr_at_fpr05,
        'actual_FPR': actual_fpr,
        'threshold_tau': threshold_tau,
        'median_residuals': median_residuals,
        'num_runs_per_label': {
            'gold': len(traces['gold']),
            'creative': len(traces['creative']),
            'drift': len(traces['drift']),
        },
        'rank_k': k,
        'real_traces': data_source == 'real_traces_only',
    }

    # Save report
    report_path = output_dir / 'validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    logger.info("  Saved report to %s", report_path)

    # Generate plots
    if 'fpr' in metrics and 'tpr' in metrics:
        plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['auc'], output_dir)
        logger.info("  Saved ROC curve")

    plot_distributions(features_df, output_dir)
    logger.info("  Saved distribution plots")

    # Log summary
    logger.info("  Results:")
    logger.info("    AUC: %.4f", report['AUC'])
    logger.info("    TPR @ FPR=0.05: %.4f", report['TPR_at_FPR05'])
    logger.info("    Threshold tau: %.4f", report['threshold_tau'])

    return report


def run_all_scenarios(
    scenarios_dir: Path,
    reports_dir: Path,
    traces_base: Path,
    k: int = 10,
) -> Dict[str, Dict[str, Any]]:
    """Run experiments for all scenarios.

    Parameters
    ----------
    scenarios_dir : Path
        Directory containing scenario subdirectories.
    reports_dir : Path
        Base directory for reports.
    traces_base : Path
        Base directory for experiment traces.
    k : int
        Rank for spectral analysis.

    Returns
    -------
    dict
        Mapping from scenario name to report dict.
    """
    all_reports = {}

    # Check for per-scenario traces
    for scenario_dir in sorted(scenarios_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue
        if scenario_dir.name.startswith('.'):
            continue

        scenario_name = scenario_dir.name

        # Look for traces in scenario-specific location first
        traces_dir = scenario_dir / 'experiment_traces'
        if not traces_dir.exists():
            # Fall back to global traces
            traces_dir = traces_base

        if not traces_dir.exists():
            logger.warning("Skipping %s: no traces found", scenario_name)
            continue

        output_dir = reports_dir / scenario_name
        report = run_experiment(
            traces_dir=traces_dir,
            output_dir=output_dir,
            k=k,
            scenario_name=scenario_name,
        )
        all_reports[scenario_name] = report

    # If no per-scenario traces, run on global traces
    if not all_reports and traces_base.exists():
        logger.info("Running on global experiment_traces...")
        report = run_experiment(
            traces_dir=traces_base,
            output_dir=reports_dir / 'global',
            k=k,
            scenario_name='global',
        )
        all_reports['global'] = report

    return all_reports


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description="Run spectral validation experiments."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Specific scenario to run (or 'all' for all scenarios)",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Run all scenarios",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Rank for spectral analysis",
    )
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=PROJECT_ROOT / "experiment_traces",
        help="Directory with experiment traces",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "spectral_validation",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("Spectral Validation Experiment Runner")
    logger.info("Using real agent traces for evaluation")
    logger.info("=" * 60)

    # Run experiments
    if args.all_scenarios or args.scenario == 'all':
        scenarios_dir = PROJECT_ROOT / "scenarios"
        reports = run_all_scenarios(
            scenarios_dir=scenarios_dir,
            reports_dir=args.output_dir,
            traces_base=args.traces_dir,
            k=args.k,
        )

        # Log summary
        logger.info("=" * 60)
        logger.info("Summary of All Scenarios")
        logger.info("=" * 60)

        all_pass = True
        for scenario_name, report in reports.items():
            auc = report.get('AUC', 0.0)
            tpr = report.get('TPR_at_FPR05', 0.0)
            status = 'PASS' if auc >= 0.90 and tpr >= 0.80 else 'WARN'
            if status == 'WARN':
                all_pass = False
            logger.info("  %s: AUC=%.4f, TPR@FPR05=%.4f [%s]", scenario_name, auc, tpr, status)

        if all_pass:
            logger.info("All scenarios PASS validation thresholds (AUC >= 0.90, TPR >= 0.80)")
        else:
            logger.warning("Some scenarios did not meet thresholds")

    elif args.scenario:
        # Run single scenario
        traces_dir = args.traces_dir
        scenario_traces = PROJECT_ROOT / "scenarios" / args.scenario / "experiment_traces"
        if scenario_traces.exists():
            traces_dir = scenario_traces

        run_experiment(
            traces_dir=traces_dir,
            output_dir=args.output_dir / args.scenario,
            k=args.k,
            scenario_name=args.scenario,
        )

    else:
        # Run on global traces
        run_experiment(
            traces_dir=args.traces_dir,
            output_dir=args.output_dir / 'global',
            k=args.k,
            scenario_name='global',
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()
