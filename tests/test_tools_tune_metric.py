#!/usr/bin/env python3
"""Tests for tools/tune_metric.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent


def test_tune_metric_writes_outputs(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    scenarios = [f"scenario_{i}" for i in range(5)]
    labels = ["coherent", "creative", "drift", "poison"]

    for scenario in scenarios:
        for idx in range(4):
            label = labels[idx % len(labels)]
            rows.append({
                "run_id": f"{scenario}_{idx}",
                "scenario": scenario,
                "label": label,
                "theoretical_bound": float(rng.normal(loc=0.5, scale=0.1)),
                "residual": float(rng.normal(loc=0.2, scale=0.05)),
                "koopman_residual": float(rng.normal(loc=0.1, scale=0.02)),
                "pca_explained": float(rng.uniform(0.7, 0.9)),
                "r_eff": float(rng.integers(1, 5)),
                "length_T": float(rng.integers(10, 20)),
                "embed_norm": float(rng.uniform(0.8, 1.2)),
                "semantic_drift": float(rng.uniform(0.1, 0.3)),
                "insample_residual": float(rng.uniform(0.1, 0.2)),
                "oos_residual": float(rng.uniform(0.1, 0.3)),
            })

    features_csv = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(features_csv, index=False)

    reports_dir = PROJECT_ROOT / "reports"
    results_path = reports_dir / "tuning_results.json"
    model_path = reports_dir / "best_model.pkl"
    if results_path.exists():
        results_path.unlink()
    if model_path.exists():
        model_path.unlink()

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "tune_metric.py"),
        "--features-csv",
        str(features_csv),
        "--groups-col",
        "scenario",
        "--label-col",
        "label",
        "--fpr-target",
        "0.05",
        "--seed",
        "0",
        "--n-boot",
        "10",
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    assert results_path.exists()
    assert model_path.exists()
    data = json.loads(results_path.read_text())
    assert "outer_folds" in data
    assert "auc_ci" in data
