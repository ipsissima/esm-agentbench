import json
from pathlib import Path
import subprocess
import os
import pytest
import yaml


def test_calibration_runs_dry(tmp_path):
    # We run calibration with a small trial count to check the script runs and writes outputs.
    repo_root = Path.cwd()
    cfg_path = repo_root / "evaluation_config.yaml"
    assert cfg_path.exists(), "evaluation_config.yaml must exist"

    # run calibration with small trials
    cmd = ["python", "tools/calibrate_thresholds.py", "--trials", "2", "--backends", "sentence-transformers", "tfidf"]
    env = {**os.environ, "ESM_FAST_EMBEDDINGS": "1"}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    print(proc.stdout)
    print(proc.stderr)
    assert proc.returncode == 0

    # check outputs
    calib1 = Path("certificates") / "calibration_sentence-transformers.json"
    calib2 = Path("certificates") / "calibration_tfidf.json"
    assert calib1.exists()
    assert calib2.exists()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert cfg["calibration"].get("residual_threshold") is not None or cfg["calibration"].get("calibrated_on") is not None
