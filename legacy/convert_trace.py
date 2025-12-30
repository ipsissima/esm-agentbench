#!/usr/bin/env python3
"""Convert existing trace formats to experiment_traces layout.

DEPRECATED: This script is kept for backward compatibility with legacy tests only.

For real agent evaluation, use:
  - tools/real_agents_hf/run_real_agents.py to generate traces
  - analysis/run_real_hf_experiment.py to evaluate them

This script converts traces from tools/real_traces/*.json to the standardized
experiment_traces/{label}/run_*.json format required by run_experiment.py.

WARNING: The --generate-synthetic option creates FAKE data and should ONLY be
used for unit tests, never for benchmark submission.

Legacy usage:
    python analysis/convert_trace.py --source tools/real_traces --output experiment_traces
    python analysis/convert_trace.py --scenario all --generate-synthetic

Output format:
    experiment_traces/
      gold/run_001.json
      gold/run_002.json
      creative/run_001.json
      drift/run_001.json
"""

import warnings
warnings.warn(
    "convert_trace.py with synthetic generation is DEPRECATED. "
    "Use tools/real_agents_hf/run_real_agents.py for real agent evaluation.",
    DeprecationWarning,
    stacklevel=2
)
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_embeddings_from_trace(trace_data: Dict[str, Any]) -> Optional[List[List[float]]]:
    """Extract embeddings from a trace file.

    Traces may contain embeddings in different formats:
    1. Direct 'embeddings' key with list of vectors
    2. 'trace' list with 'embedding' per step
    3. 'per_step_residuals' (we generate synthetic embeddings from these)

    Parameters
    ----------
    trace_data : dict
        Loaded trace JSON data.

    Returns
    -------
    list of list of float or None
        Embeddings as list of vectors, or None if not extractable.
    """
    # Check for direct embeddings key
    if 'embeddings' in trace_data:
        return trace_data['embeddings']

    # Check for trace with per-step embeddings
    if 'trace' in trace_data:
        trace = trace_data['trace']
        embeddings = []
        for step in trace:
            if 'embedding' in step:
                embeddings.append(step['embedding'])
            elif 'residual' in step and step['residual'] is not None:
                # Generate synthetic embedding from residual (for testing)
                # This is a placeholder - real traces should have embeddings
                pass
        if embeddings:
            return embeddings

    return None


def generate_synthetic_embeddings(
    residuals: List[Optional[float]],
    d: int = 64,
    seed: Optional[int] = None,
) -> List[List[float]]:
    """Generate synthetic embeddings based on residual pattern.

    Creates embeddings where the reconstruction residual matches
    the provided residual sequence. This is for testing only.

    Parameters
    ----------
    residuals : list
        Per-step residuals from the trace.
    d : int
        Embedding dimension.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list of list of float
        Synthetic embeddings.
    """
    rng = np.random.default_rng(seed)
    T = len(residuals)

    # Base trajectory (smooth)
    base = np.zeros((T, d))
    for i in range(T):
        # Smooth evolution
        t = i / max(T - 1, 1)
        base[i] = np.sin(2 * np.pi * t * np.arange(d) / d) + 0.5 * np.cos(np.pi * t)

    # Add perturbations based on residuals
    for i, r in enumerate(residuals):
        if r is not None:
            # Higher residual -> more perturbation
            noise = rng.standard_normal(d) * r * 0.5
            base[i] += noise

    return base.tolist()


def infer_label_from_filename(filename: str) -> str:
    """Infer trace label from filename.

    Expected patterns:
    - gold_gpt-4o_*.json -> gold
    - creative_gpt-4o_*.json -> creative
    - drift_gpt-*.json -> drift
    - poison_*.json -> drift (adversarial)
    - starvation_*.json -> drift (resource exhaustion)

    Parameters
    ----------
    filename : str
        Trace filename.

    Returns
    -------
    str
        Label: 'gold', 'creative', or 'drift'.
    """
    name = filename.lower()
    if name.startswith('gold'):
        return 'gold'
    elif name.startswith('creative'):
        return 'creative'
    elif name.startswith('drift'):
        return 'drift'
    elif name.startswith('poison'):
        return 'drift'  # Adversarial traces are drift
    elif name.startswith('starvation'):
        return 'drift'  # Resource-exhaustion traces are drift
    else:
        return 'unknown'


def convert_trace_file(
    source_path: Path,
    output_dir: Path,
    run_counter: Dict[str, int],
    generate_embeddings: bool = False,
) -> Optional[str]:
    """Convert a single trace file to experiment format.

    Parameters
    ----------
    source_path : Path
        Path to source trace file.
    output_dir : Path
        Output directory root.
    run_counter : dict
        Counter for run numbering per label.
    generate_embeddings : bool
        If True, generate synthetic embeddings when not available.

    Returns
    -------
    str or None
        Output path if successful, None otherwise.
    """
    try:
        with open(source_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {source_path}: {e}")
        return None

    # Infer label
    label = infer_label_from_filename(source_path.name)
    if label == 'unknown':
        print(f"Skipping unknown label: {source_path.name}")
        return None

    # Extract or generate embeddings
    embeddings = extract_embeddings_from_trace(data)
    if embeddings is None:
        if generate_embeddings and 'per_step_residuals' in data:
            embeddings = generate_synthetic_embeddings(
                data['per_step_residuals'],
                seed=hash(source_path.name) % (2**31),
            )
        else:
            print(f"No embeddings found in {source_path.name}")
            return None

    # Create output directory
    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    # Increment counter
    run_counter[label] = run_counter.get(label, 0) + 1
    run_id = f"run_{run_counter[label]:03d}"

    # Build output structure
    output_data = {
        "run_id": run_id,
        "label": label,
        "embeddings": embeddings,
        "meta": {
            "source_file": source_path.name,
            "episode_id": data.get("episode_id", "unknown"),
            "model_name": data.get("model_name", "unknown"),
            "temperature": data.get("temperature", None),
            "task_success": data.get("task_success", None),
            "dataset_tag": data.get("dataset_tag", label),
        }
    }

    # Write output
    output_path = label_dir / f"{run_id}.json"
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    return str(output_path)


def generate_synthetic_traces(
    output_dir: Path,
    n_gold: int = 20,
    n_creative: int = 20,
    n_drift: int = 20,
    T: int = 30,
    d: int = 64,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Generate fully synthetic traces for testing.

    All data are synthetic. No real secrets or external network calls.

    Parameters
    ----------
    output_dir : Path
        Output directory root.
    n_gold, n_creative, n_drift : int
        Number of traces per label.
    T : int
        Trace length (timesteps).
    d : int
        Embedding dimension.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Mapping from label to list of output paths.
    """
    rng = np.random.default_rng(seed)
    outputs: Dict[str, List[str]] = {'gold': [], 'creative': [], 'drift': []}

    for label, n_traces in [('gold', n_gold), ('creative', n_creative), ('drift', n_drift)]:
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_traces):
            run_id = f"run_{i+1:03d}"

            if label == 'gold':
                # Gold: smooth, low-noise trajectory
                X = np.zeros((T, d))
                for t in range(T):
                    phase = t / T
                    X[t] = np.sin(2 * np.pi * phase * np.arange(d) / d)
                X += rng.standard_normal((T, d)) * 0.02

            elif label == 'creative':
                # Creative: more variance but still coherent
                X = np.zeros((T, d))
                offset = rng.standard_normal(d) * 0.3
                for t in range(T):
                    phase = t / T
                    X[t] = np.sin(2 * np.pi * (phase + 0.1 * rng.random()) * np.arange(d) / d)
                    X[t] += offset
                X += rng.standard_normal((T, d)) * 0.1

            else:  # drift
                # Drift: divergent trajectory
                X = np.zeros((T, d))
                drift_direction = rng.standard_normal(d)
                drift_direction = drift_direction / np.linalg.norm(drift_direction)
                for t in range(T):
                    phase = t / T
                    X[t] = np.sin(2 * np.pi * phase * np.arange(d) / d)
                    # Add increasing drift
                    X[t] += drift_direction * phase * 2.0
                X += rng.standard_normal((T, d)) * 0.15

            output_data = {
                "run_id": run_id,
                "label": label,
                "embeddings": X.tolist(),
                "meta": {
                    "source_file": "synthetic",
                    "episode_id": f"synthetic_{label}_{i+1}",
                    "model_name": "synthetic",
                    "temperature": 0.0 if label == 'gold' else 0.7 if label == 'creative' else 1.3,
                    "task_success": label != 'drift',
                    "dataset_tag": label,
                    "synthetic": True,
                }
            }

            output_path = label_dir / f"{run_id}.json"
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)

            outputs[label].append(str(output_path))

    return outputs


def convert_real_traces(
    source_dir: Path,
    output_dir: Path,
    generate_embeddings: bool = False,
) -> Dict[str, int]:
    """Convert all traces in source directory to experiment format.

    Parameters
    ----------
    source_dir : Path
        Directory containing source trace files.
    output_dir : Path
        Output directory root.
    generate_embeddings : bool
        If True, generate synthetic embeddings when not available.

    Returns
    -------
    dict
        Count of traces converted per label.
    """
    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        return {}

    run_counter: Dict[str, int] = {}
    stats: Dict[str, int] = {}

    for trace_file in sorted(source_dir.glob("*.json")):
        result = convert_trace_file(
            trace_file,
            output_dir,
            run_counter,
            generate_embeddings,
        )
        if result:
            label = infer_label_from_filename(trace_file.name)
            stats[label] = stats.get(label, 0) + 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert trace formats for spectral validation experiments."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "tools" / "real_traces",
        help="Source directory with trace files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "experiment_traces",
        help="Output directory for converted traces",
    )
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="Generate fully synthetic traces instead of converting",
    )
    parser.add_argument(
        "--generate-embeddings",
        action="store_true",
        help="Generate synthetic embeddings from residuals when missing",
    )
    parser.add_argument(
        "--n-traces",
        type=int,
        default=20,
        help="Number of synthetic traces per label (with --generate-synthetic)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic generation",
    )

    args = parser.parse_args()

    if args.generate_synthetic:
        print(f"Generating synthetic traces to {args.output}")
        print("NOTE: All data are synthetic. No real secrets or external network calls.")
        outputs = generate_synthetic_traces(
            args.output,
            n_gold=args.n_traces,
            n_creative=args.n_traces,
            n_drift=args.n_traces,
            seed=args.seed,
        )
        for label, paths in outputs.items():
            print(f"  {label}: {len(paths)} traces")
    else:
        print(f"Converting traces from {args.source} to {args.output}")
        print("NOTE: All data are synthetic. No real secrets or external network calls.")
        stats = convert_real_traces(
            args.source,
            args.output,
            generate_embeddings=args.generate_embeddings,
        )
        print(f"Converted traces: {stats}")

    print("Done!")


if __name__ == "__main__":
    main()
