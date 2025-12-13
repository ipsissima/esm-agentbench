"""Synthetic HaluEval benchmark adapter.

This script builds mock traces to demonstrate how the spectral certificate
responds to coherent versus hallucinated content. Replace the ``load_halueval_sample``
function with a real dataset loader to run against HaluEval.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

# Enable imports from the repository root when executed directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from benchmarks.utils import evaluate_dataset, print_metrics


def load_halueval_sample() -> List[Dict[str, List[Dict[str, str]]]]:
    """Generate synthetic HaluEval-style samples.

    Each sample consists of a trace with multiple reasoning steps and a
    binary label: ``1`` for hallucinated (bad) and ``0`` for truthful (good).
    """

    good_trace = [
        {"text": "Question: What is the capital of France?"},
        {"text": "Answer: The capital of France is Paris."},
        {"text": "Answer: The capital of France is Paris."},
        {"text": "Answer: The capital of France is Paris."},
        {"text": "Answer: The capital of France is Paris."},
    ]

    bad_trace = [
        {"text": "Question: What is the capital of France?"},
        {"text": "Reasoning: Maybe it's Rome because of the Roman Empire."},
        {"text": "Digression: Rome is known for pasta and ancient arenas."},
        {"text": "Shift: Actually, Paris sounds like a dessert more than a city."},
        {"text": "Answer: Therefore, the capital of France is Rome."},
    ]

    samples: List[Dict[str, object]] = []
    for i in range(25):
        samples.append({"trace": good_trace, "label": 0})
        samples.append({"trace": bad_trace, "label": 1})

    return samples


def main() -> None:
    samples = load_halueval_sample()
    results = evaluate_dataset(samples)
    print("HaluEval Benchmark Results:")
    print_metrics(results)


if __name__ == "__main__":  # pragma: no cover
    main()
