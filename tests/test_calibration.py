from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def test_calibration_dry_run(tmp_path):
    env = os.environ.copy()
    env["ESM_FORCE_TFIDF"] = "1"
    calib_path = Path("certificates/calibration_sentence-transformers.json")
    cfg_path = Path("evaluation_config.yaml")
    original_cfg = cfg_path.read_text(encoding="utf-8")
    if calib_path.exists():
        calib_path.unlink()
    try:
        subprocess.check_call([
            "python",
            "tools/calibrate_thresholds.py",
            "--backend",
            "sentence-transformers",
            "--trials",
            "2",
            "--dry-run",
        ], env=env)
        assert calib_path.exists()
        data = json.loads(calib_path.read_text())
        assert "threshold" in data
    finally:
        calib_path.unlink(missing_ok=True)
        cfg_path.write_text(original_cfg, encoding="utf-8")
