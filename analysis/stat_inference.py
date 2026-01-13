"""Statistical inference utilities for evaluation metrics."""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, Tuple

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_SKLEARN = False


def bootstrap_auc(
    y_true: Iterable[int],
    y_score: Iterable[float],
    n_boot: int = 1000,
) -> Tuple[float, float]:
    """Compute a 95% bootstrap confidence interval for AUC.

    Parameters
    ----------
    y_true : iterable of int
        Ground-truth binary labels.
    y_score : iterable of float
        Model scores or probabilities.
    n_boot : int, optional
        Number of bootstrap resamples.

    Returns
    -------
    tuple of float
        Lower and upper bounds of the 95% confidence interval.
    """
    if not HAS_SKLEARN:
        return float("nan"), float("nan")

    y_true_arr = np.asarray(list(y_true))
    y_score_arr = np.asarray(list(y_score))

    if y_true_arr.size == 0 or y_true_arr.size != y_score_arr.size:
        return float("nan"), float("nan")

    rng = np.random.default_rng(0)
    auc_samples = []
    for _ in range(n_boot):
        indices = rng.integers(0, y_true_arr.size, size=y_true_arr.size)
        sample_true = y_true_arr[indices]
        sample_score = y_score_arr[indices]
        if np.unique(sample_true).size < 2:
            continue
        auc_samples.append(roc_auc_score(sample_true, sample_score))

    if not auc_samples:
        return float("nan"), float("nan")

    lower, upper = np.percentile(auc_samples, [2.5, 97.5])
    return float(lower), float(upper)


def permutation_test_score(
    group_a: Iterable[float],
    group_b: Iterable[float],
) -> float:
    """Exact permutation test for separation between two distributions.

    Parameters
    ----------
    group_a : iterable of float
        Scores from group A (e.g., Gold).
    group_b : iterable of float
        Scores from group B (e.g., Drift).

    Returns
    -------
    float
        Two-sided exact p-value based on difference in means.
    """
    a = np.asarray(list(group_a), dtype=float)
    b = np.asarray(list(group_b), dtype=float)

    if a.size == 0 or b.size == 0:
        return float("nan")

    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n_a = a.size

    count = 0
    total = 0
    for indices in combinations(range(pooled.size), n_a):
        indices = np.asarray(indices)
        sample_a = pooled[indices]
        sample_b = np.delete(pooled, indices)
        stat = abs(sample_a.mean() - sample_b.mean())
        if stat >= observed:
            count += 1
        total += 1

    return float(count / total)
