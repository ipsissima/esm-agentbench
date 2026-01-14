#!/usr/bin/env python3
"""Power analysis for drift detection metrics."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from tools.feature_utils import PROJECT_ROOT, label_to_binary

logger = logging.getLogger(__name__)


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return 0.0
    pooled = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled)


def required_sample_size(d: float, alpha: float = 0.05, power: float = 0.8) -> float:
    if d <= 0:
        return float("inf")
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    n_per_group = 2 * (z_alpha + z_beta) ** 2 / (d ** 2)
    return float(np.ceil(n_per_group))


def bootstrap_cohen_d(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        x_idx = rng.integers(0, len(x), size=len(x))
        y_idx = rng.integers(0, len(y), size=len(y))
        vals.append(cohen_d(x[x_idx], y[y_idx]))
    low, high = np.percentile(vals, [2.5, 97.5])
    return float(low), float(high)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Power analysis for drift detection features.")
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000)

    args = parser.parse_args()

    df = pd.read_csv(args.features_csv)
    if "label" not in df.columns:
        raise ValueError("features CSV must include label column")

    df = df.copy()
    df["label_binary"] = df["label"].apply(label_to_binary)
    df = df.dropna(subset=["label_binary"])

    results = []
    for scenario, group in df.groupby("scenario") if "scenario" in df.columns else [("global", df)]:
        good = group[group["label_binary"] == 0]["theoretical_bound"].to_numpy()
        bad = group[group["label_binary"] == 1]["theoretical_bound"].to_numpy()
        if len(good) < 2 or len(bad) < 2:
            continue
        d = cohen_d(bad, good)
        ci_low, ci_high = bootstrap_cohen_d(bad, good, n_boot=args.n_boot, seed=args.seed)
        n_required = required_sample_size(abs(d), alpha=args.alpha, power=args.power)

        results.append({
            "scenario": scenario,
            "effect_size_d": d,
            "d_ci_low": ci_low,
            "d_ci_high": ci_high,
            "alpha": args.alpha,
            "power": args.power,
            "n_per_group": n_required,
        })

    output_path = PROJECT_ROOT / "reports" / "power_analysis.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    logger.info("Saved power analysis to %s", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
