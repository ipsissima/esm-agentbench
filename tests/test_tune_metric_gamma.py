import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def test_tune_metric_gamma_grid(tmp_path):
    features = pd.DataFrame(
        {
            "run_id": [f"run{i}" for i in range(8)],
            "scenario": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
            "label": ["coherent", "drift", "coherent", "drift", "coherent", "drift", "coherent", "drift"],
            "theoretical_bound": np.linspace(0.1, 0.8, 8),
            "theoretical_bound_norm": np.linspace(0.1, 0.8, 8),
            "residual": np.linspace(0.2, 0.9, 8),
            "residual_norm": np.linspace(0.2, 0.9, 8),
            "residual_fro_norm": np.linspace(0.1, 0.4, 8),
            "koopman_residual": np.linspace(0.1, 0.8, 8),
            "pca_explained": np.linspace(0.2, 0.9, 8),
            "r_eff": np.linspace(2, 9, 8),
            "r_rel": np.linspace(0.1, 0.8, 8),
            "length_T": np.repeat(10, 8),
            "embed_norm": np.linspace(0.5, 1.2, 8),
            "semantic_drift": np.linspace(0.1, 0.8, 8),
            "semantic_centroid_distance": np.linspace(0.0, 0.7, 8),
            "sv_max_ratio": np.linspace(0.1, 0.8, 8),
            "insample_residual": np.linspace(0.2, 0.9, 8),
            "oos_residual": np.linspace(0.2, 0.9, 8),
        }
    )
    features_path = tmp_path / "features.csv"
    features.to_csv(features_path, index=False)

    reports_dir = Path("reports")
    if reports_dir.exists():
        for path in reports_dir.glob("tuning_results.json"):
            path.unlink()

    cmd = (
        f"python tools/tune_metric.py --features-csv {features_path} "
        "--gamma-grid 0 0.2 0.4 --use-prompt-adj --n-boot 10 --fpr-target 0.5"
    )
    assert os.system(cmd) == 0

    results_path = Path("reports") / "tuning_results.json"
    assert results_path.exists()
    data = json.loads(results_path.read_text())
    assert "best_gamma" in data
