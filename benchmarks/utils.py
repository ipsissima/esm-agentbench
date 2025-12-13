"""Utilities for running benchmark adapters.

This module provides a thin wrapper around the existing embedding and
certificate logic. It expects samples in the form:
```
{"trace": [{"text": "step 1"}, {"text": "step 2"}, ...], "label": 0 or 1}
```
where ``label`` is ``1`` for hallucinated/bad traces and ``0`` for
truthful/good traces.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

# Skip heavyweight embedding downloads to keep benchmarks snappy in offline mode.
os.environ.setdefault("SKIP_SENTENCE_TRANSFORMERS", "1")

# Ensure local imports work when invoked from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from assessor.kickoff import embed_trace_steps
from certificates.make_certificate import compute_certificate


def evaluate_dataset(samples: Sequence[dict]) -> List[Tuple[float, int]]:
    """Compute spectral certificate bounds for a dataset.

    Args:
        samples: Iterable of mapping objects with ``trace`` and ``label`` keys.

    Returns:
        List of ``(theoretical_bound, label)`` tuples in input order.
    """

    results: List[Tuple[float, int]] = []
    for sample in samples:
        trace = sample.get("trace", [])
        label = int(sample.get("label", 0))

        embeddings = embed_trace_steps(trace)
        certificate = compute_certificate(embeddings)
        bound = float(certificate.get("theoretical_bound", np.nan))

        results.append((bound, label))

    return results


def print_metrics(results: Iterable[Tuple[float, int]]) -> None:
    """Print AUC-ROC and per-class average bounds."""

    bounds, labels = zip(*results)
    labels_arr = np.asarray(labels)
    bounds_arr = np.asarray(bounds, dtype=float)

    auc = roc_auc_score(labels_arr, bounds_arr)

    good_mask = labels_arr == 0
    bad_mask = labels_arr == 1
    good_avg = float(np.nanmean(bounds_arr[good_mask])) if good_mask.any() else float("nan")
    bad_avg = float(np.nanmean(bounds_arr[bad_mask])) if bad_mask.any() else float("nan")

    print(f"AUC-ROC: {auc:.3f}")
    print(f"Average bound (Good=0): {good_avg:.4f}")
    print(f"Average bound (Bad=1):  {bad_avg:.4f}")
