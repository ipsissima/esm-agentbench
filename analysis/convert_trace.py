#!/usr/bin/env python3
"""
Minimal synthetic trace generator for spectral validation.

Creates experiment_traces/{gold,creative,drift}/run_XXX.json files with fields:
  - run_id: string
  - label: one of 'gold', 'creative', 'drift'
  - embeddings: list[list[float]]  # T x D

Usage:
  python analysis/convert_trace.py --generate-synthetic --n-traces 30 --seed 42
"""

import argparse
import json
import os
from pathlib import Path
from typing import List
import numpy as np

DEFAULT_OUT = Path("experiment_traces")

def make_trace(run_id: str, label: str, T: int, D: int, rng: np.random.Generator) -> dict:
    # baseline signal for gold
    if label == "gold":
        base = rng.normal(loc=0.0, scale=0.5, size=(T, D))
    elif label == "creative":
        # creative: larger variance, same mean
        base = rng.normal(loc=0.0, scale=1.2, size=(T, D))
        # optionally add random low-frequency trend
        trend = np.linspace(0, 0.5, T).reshape(T, 1) * rng.normal(scale=0.2, size=(1, D))
        base += trend
    else:  # drift
        # drift: mean shift over time
        base = rng.normal(loc=0.0, scale=0.6, size=(T, D))
        shift = np.linspace(0, 1.5, T).reshape(T, 1)
        base += shift

    # Clip or convert to python floats
    embeddings: List[List[float]] = base.tolist()
    return {"run_id": run_id, "label": label, "embeddings": embeddings}

def generate_synthetic(n_traces: int = 30, seed: int = 42, out_dir: Path = DEFAULT_OUT):
    rng = np.random.default_rng(seed)
    D = 16  # embedding dimension
    # split evenly into 3 categories
    n_each = max(1, n_traces // 3)
    counts = {"gold": n_each, "creative": n_each, "drift": n_traces - 2 * n_each}

    # ensure out dirs exist
    for lab in counts:
        (out_dir / lab).mkdir(parents=True, exist_ok=True)

    idx = 0
    for lab, cnt in counts.items():
        for i in range(cnt):
            idx += 1
            run_id = f"run_{idx:03d}"
            # random length between 8 and 40
            T = int(rng.integers(8, 40))
            trace = make_trace(run_id, lab, T, D, rng)
            out_path = out_dir / lab / f"{run_id}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(trace, f)
    print(f"Generated synthetic traces ({n_traces}) under {out_dir}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generate-synthetic", action="store_true")
    p.add_argument("--n-traces", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    args = p.parse_args()

    if args.generate_synthetic:
        generate_synthetic(n_traces=args.n_traces, seed=args.seed, out_dir=Path(args.out_dir))
    else:
        print("No action. Use --generate-synthetic to produce synthetic traces.")

if __name__ == "__main__":
    main()
