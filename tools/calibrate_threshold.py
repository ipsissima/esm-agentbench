"""Calibrate residual thresholds from existing gold demo traces."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACE_DIR = REPO_ROOT / "demo_traces"
CONFIG_DIR = REPO_ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "calibration.json"


def _collect_residuals() -> List[float]:
    residuals: List[float] = []
    if not TRACE_DIR.exists():
        return residuals
    for path in TRACE_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("dataset_tag") != "gold":
            continue
        per_step = data.get("per_step_residuals") or []
        if not per_step and isinstance(data.get("trace"), list):
            per_step = [entry.get("residual") for entry in data["trace"] if isinstance(entry, dict)]
        for val in per_step:
            if val is None:
                continue
            try:
                residuals.append(float(val))
            except Exception:
                continue
    return residuals


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as tmp:
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def main() -> None:
    residuals = _collect_residuals()
    if residuals:
        threshold = float(np.percentile(residuals, 95))
    else:
        threshold = 0.0
    config = {
        "residual_threshold": threshold,
        "pivot_spike_factor": 5.0,
        "post_stabilize_factor": 0.5,
    }
    _atomic_write(CONFIG_PATH, config)

    print("metric,value")
    print(f"count,{len(residuals)}")
    print(f"percentile_95,{threshold:.6f}")


if __name__ == "__main__":  # pragma: no cover
    main()
