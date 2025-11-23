"""Utility to calibrate residual thresholds from stored demo traces."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACE_DIR = REPO_ROOT / "demo_traces"
CONFIG_PATH = REPO_ROOT / "config" / "calibration.json"
SUMMARY_PATH = REPO_ROOT / "demo_traces" / "calibration_summary.csv"


def _load_residuals(tag: str) -> List[float]:
    vals: List[float] = []
    for path in TRACE_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("dataset_tag") != tag:
            continue
        for resid in data.get("per_step_residuals", []):
            if resid is not None:
                try:
                    vals.append(float(resid))
                except Exception:
                    continue
    return vals


def main(tag: str = "gold") -> None:
    residuals = _load_residuals(tag)
    if not residuals:
        raise SystemExit("No residuals found for calibration")

    threshold = float(np.percentile(residuals, 95))
    pivot_spike_factor = 4.0
    post_stabilize_factor = 0.5

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(
            {
                "residual_threshold": threshold,
                "pivot_spike_factor": pivot_spike_factor,
                "post_stabilize_factor": post_stabilize_factor,
                "source_tag": tag,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "residual": residuals,
        }
    )
    df.describe(percentiles=[0.5, 0.75, 0.9, 0.95]).to_csv(SUMMARY_PATH)
    print(
        f"Calibration complete: threshold={threshold:.4f}, pivot_spike_factor={pivot_spike_factor}, "
        f"post_stabilize_factor={post_stabilize_factor}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
