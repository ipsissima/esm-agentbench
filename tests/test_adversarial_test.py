#!/usr/bin/env python3
"""Tests for tools/adversarial_test.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tools.feature_utils import compute_trace_features

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.mark.xfail(
    reason="Kernel SIGSEGV: verified kernel crashes during certificate computation",
    strict=False,
)
def test_adversarial_script_runs(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()

    trace_paths = []
    for idx, label in enumerate(["coherent", "drift"]):
        embeddings = rng.normal(size=(6, 4)).tolist()
        trace_path = traces_dir / f"trace_{idx}.json"
        trace_path.write_text(json.dumps({"embeddings": embeddings, "label": label}))
        trace_paths.append(trace_path)

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

    rows = []
    labels = []
    for trace_path in trace_paths:
        data = json.loads(trace_path.read_text())
        embeddings = np.asarray(data["embeddings"], dtype=np.float64)
        features = compute_trace_features(embeddings)
        rows.append({col: features.get(col, 0.0) for col in feature_cols})
        labels.append(1 if data["label"] == "drift" else 0)

    X = pd.DataFrame(rows)
    y = np.array(labels)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=0)),
    ])
    model.fit(X, y)

    model_path = tmp_path / "model.pkl"
    joblib.dump({"model": model, "feature_columns": feature_cols}, model_path)

    reports_dir = PROJECT_ROOT / "reports"
    summary_path = reports_dir / "adversarial_summary.json"
    if summary_path.exists():
        summary_path.unlink()

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "adversarial_test.py"),
        "--model",
        str(model_path),
        "--traces-dir",
        str(traces_dir),
        "--attack",
        "finite-diff",
        "--budget",
        "5",
        "--seed",
        "0",
        "--norm-clip",
        "0.1",
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    assert summary_path.exists()
