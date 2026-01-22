#!/usr/bin/env python3
"""Ensure no global RNG seeding in evidence tools."""
from __future__ import annotations

from pathlib import Path

import pytest

# Mark all tests in this module as unit tests (Tier 1)
pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).parent.parent

CHECK_FILES = [
    PROJECT_ROOT / "analysis" / "run_experiment.py",
    PROJECT_ROOT / "tools" / "tune_metric.py",
    PROJECT_ROOT / "tools" / "eval_holdout.py",
    PROJECT_ROOT / "tools" / "adversarial_test.py",
    PROJECT_ROOT / "tools" / "feature_utils.py",
    PROJECT_ROOT / "tools" / "power_analysis.py",
]

BANNED_PATTERNS = ["np.random.seed", "random.seed("]


def test_no_global_rng_seeding() -> None:
    violations = []
    for path in CHECK_FILES:
        if not path.exists():
            continue
        content = path.read_text()
        for pattern in BANNED_PATTERNS:
            if pattern in content:
                violations.append((path, pattern))

    if violations:
        formatted = "\n".join([f"{path}: {pattern}" for path, pattern in violations])
        pytest.fail(f"Global RNG seeding found in evidence paths:\n{formatted}")
