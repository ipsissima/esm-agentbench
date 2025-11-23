#!/usr/bin/env python
"""Calibrate residual thresholds from cached traces."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from assessor.kickoff import embed_trace_steps, _compute_residuals, run_episode


def load_traces(trace_dir: Path) -> List[List[dict]]:
    traces: List[List[dict]] = []
    for path in trace_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                traces.append(data)
        except Exception:
            continue
    return traces


def generate_deterministic_samples(count: int = 2) -> List[List[dict]]:
    samples = []
    for _ in range(count):
        ep = run_episode({"prompt": "Compute Fibonacci with CoT steps", "max_steps": 8})
        samples.append(ep["trace"])
    return samples


def compute_residuals_for_traces(traces: List[List[dict]]) -> List[float]:
    residuals: List[float] = []
    for trace in traces:
        embeddings = embed_trace_steps(trace)
        res, _, _ = _compute_residuals(embeddings, threshold=1e9)
        residuals.extend([r for r in res if r is not None])
    return residuals


def main():
    parser = argparse.ArgumentParser(description="Calibrate Koopman residual thresholds from traces")
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("tools/real_traces"),
        help="Directory containing trace JSON files.",
    )
    args = parser.parse_args()

    traces = load_traces(args.trace_dir)
    if not traces:
        traces = generate_deterministic_samples()

    residuals = compute_residuals_for_traces(traces)
    if not residuals:
        print("No residuals computed; aborting calibration.")
        return

    percentiles = np.percentile(residuals, [50, 75, 90, 95, 99])
    residual_threshold = float(percentiles[-2])
    median_resid = float(percentiles[0])
    pivot_spike_factor = float(max(3.0, (percentiles[-1] / (median_resid + 1e-6))))

    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / "calibration.json"
    config = {
        "residual_threshold": residual_threshold,
        "pivot_spike_factor": pivot_spike_factor,
        "median_residual": median_resid,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    csv_path = config_dir / "residuals.csv"
    csv_path.write_text("\n".join(str(r) for r in residuals), encoding="utf-8")

    print("Calibrated thresholds saved to", config_path)
    print("Residual threshold (95th pct):", residual_threshold)
    print("Pivot spike factor (relative to median):", pivot_spike_factor)


if __name__ == "__main__":
    main()
