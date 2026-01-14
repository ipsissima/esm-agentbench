"""Statistical inference utilities for spectral validation."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample


def _rng_from_seed(seed: int) -> np.random.Generator:
    """Create a deterministic numpy RNG without using default_rng in analysis/."""
    return np.random.Generator(np.random.PCG64(seed))


def bootstrap_auc(
    y_true: Iterable[int], y_score: Iterable[float], n_boot: int = 1000
) -> Tuple[float, float]:
    """Compute a bootstrap 95% confidence interval for ROC AUC."""
    y_true_arr = np.asarray(list(y_true))
    y_score_arr = np.asarray(list(y_score))
    if y_true_arr.size == 0 or y_true_arr.size != y_score_arr.size:
        return (float("nan"), float("nan"))

    rng = _rng_from_seed(0)
    boot_scores = []
    indices = np.arange(y_true_arr.size)
    for _ in range(n_boot):
        sample_idx = resample(indices, replace=True, n_samples=len(indices), random_state=rng.integers(1 << 32))
        sample_y = y_true_arr[sample_idx]
        sample_score = y_score_arr[sample_idx]
        if len(np.unique(sample_y)) < 2:
            continue
        boot_scores.append(roc_auc_score(sample_y, sample_score))

    if not boot_scores:
        return (float("nan"), float("nan"))

    low, high = np.percentile(boot_scores, [2.5, 97.5])
    return (float(low), float(high))


def permutation_test_score(
    group_a: Iterable[float], group_b: Iterable[float], n_perm: int = 1000
) -> float:
    """Compute a one-sided permutation p-value for mean(group_a) > mean(group_b)."""
    a = np.asarray(list(group_a), dtype=float)
    b = np.asarray(list(group_b), dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")

    rng = _rng_from_seed(0)
    observed = float(np.mean(a) - np.mean(b))
    combined = np.concatenate([a, b])
    n_a = a.size
    count = 0

    for _ in range(n_perm):
        rng.shuffle(combined)
        diff = float(np.mean(combined[:n_a]) - np.mean(combined[n_a:]))
        if diff >= observed:
            count += 1

    return (count + 1) / (n_perm + 1)
