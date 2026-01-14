#!/usr/bin/env python3
"""
Developer-only synthetic trace generator.

THIS FILE IS FOR DEV/CI TESTS ONLY. It MUST NOT be placed under analysis/ because
analysis/ is reserved for evidence generation and must not include synthetic
trace producers.

Usage (developer CI):
  ALLOW_SYNTHETIC_TRACES=1 python tools/dev/generate_synthetic_traces.py --n 30 --seed 42 --out-dir dev_artifacts/experiment_traces
"""
from pathlib import Path
import argparse
import json

import numpy as np


def make_trace(run_id: str, label: str, T: int, D: int, rng: np.random.Generator):
    base = rng.normal(loc=0.0, scale=0.5, size=(T, D))
    return {
        "run_id": run_id,
        "label": label,
        "embeddings": base.tolist(),
        "metadata": {"data_source": "synthetic", "generator": "tools/dev/generate_synthetic_traces.py"},
    }


def generate_synthetic(n: int, seed: int, out_dir: Path):
    rng = np.random.default_rng(seed)
    D = 16
    counts = {"gold": max(1, n // 3), "creative": max(1, n // 3), "drift": n - 2 * (n // 3)}
    out_dir = Path(out_dir)
    for label, cnt in counts.items():
        (out_dir / label).mkdir(parents=True, exist_ok=True)
        for i in range(cnt):
            T = int(rng.integers(8, 40))
            run_id = f"{label}_{i:03d}"
            trace = make_trace(run_id, label, T, D, rng)
            with open(out_dir / label / f"{run_id}.json", "w", encoding="utf-8") as f:
                json.dump(trace, f)
    print(f"Generated synthetic traces: n={n} seed={seed} -> {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="dev_artifacts/experiment_traces")
    args = p.parse_args()
    generate_synthetic(args.n, args.seed, Path(args.out_dir))


if __name__ == "__main__":
    main()
