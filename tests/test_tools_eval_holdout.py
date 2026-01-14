#!/usr/bin/env python3
"""Tests for tools/eval_holdout.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent


def test_eval_holdout_outputs(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    rows = []
    for idx in range(12):
        label = "drift" if idx % 2 == 0 else "coherent"
        rows.append({
            "run_id": f"run_{idx}",
            "scenario": "scenario_a",
            "label": label,
            "theoretical_bound": float(rng.normal()),
            "residual": float(rng.normal()),
            "koopman_residual": float(rng.normal()),
            "pca_explained": float(rng.uniform(0.6, 0.9)),
            "r_eff": float(rng.integers(1, 5)),
            "length_T": float(rng.integers(10, 20)),
            "embed_norm": float(rng.uniform(0.8, 1.2)),
            "semantic_drift": float(rng.uniform(0.1, 0.3)),
            "insample_residual": float(rng.uniform(0.1, 0.2)),
            "oos_residual": float(rng.uniform(0.1, 0.3)),
        })

    features_csv = tmp_path / "holdout.csv"
    df = pd.DataFrame(rows)
    df.to_csv(features_csv, index=False)

    feature_cols = [
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
    X = df[feature_cols].fillna(0.0)
    y = np.array([1 if v == "drift" else 0 for v in df["label"]])

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=0)),
    ])
    model.fit(X, y)

    model_path = tmp_path / "model.pkl"
    joblib.dump({"model": model, "feature_columns": feature_cols}, model_path)

    reports_dir = PROJECT_ROOT / "reports"
    output_path = reports_dir / "holdout_evaluation.json"
    if output_path.exists():
        output_path.unlink()

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "eval_holdout.py"),
        "--model",
        str(model_path),
        "--features-csv",
        str(features_csv),
        "--fpr-target",
        "0.05",
        "--n-boot",
        "10",
        "--seed",
        "0",
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    assert output_path.exists()
    payload = json.loads(output_path.read_text())
    assert "metrics" in payload
    assert "auc_ci" in payload["metrics"]
