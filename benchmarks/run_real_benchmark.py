"""Run spectral certificate benchmarks on real datasets or bundled samples.

This adapter prioritizes real validation splits (HaluEval or TruthfulQA) when
available and falls back to bundled sample traces for offline CI environments.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:  # CI-friendly: the dependency might be missing
    from datasets import load_dataset  # type: ignore
except ImportError:  # pragma: no cover - exercised indirectly in CI
    load_dataset = None  # type: ignore

from benchmarks.utils import evaluate_dataset, print_metrics


def _load_halueval() -> List[Dict[str, object]]:
    """Load the HaluEval validation split into benchmark samples."""

    if load_dataset is None:
        print("The `datasets` package is required to load HaluEval. Please install it via `pip install datasets`.")
        return []

    try:
        dataset = load_dataset("halueval", split="validation")
    except Exception as exc:  # pragma: no cover - network/dependency failures
        print(f"Failed to load HaluEval: {exc}")
        return []

    samples: List[Dict[str, object]] = []
    for row in dataset:
        question = str(row.get("question") or row.get("query") or "").strip()
        answer = str(row.get("answer") or row.get("response") or row.get("output") or "").strip()

        raw_label = row.get("label") if "label" in row else row.get("hallucination")
        label = 1 if str(raw_label) == "1" or raw_label is True else 0

        trace = [
            {"text": f"Question: {question}"},
            {"text": f"Answer: {answer}"},
        ]
        samples.append({"trace": trace, "label": label})

    return samples


def _load_truthfulqa() -> List[Dict[str, object]]:
    """Load the TruthfulQA validation split into benchmark samples."""

    if load_dataset is None:
        print("The `datasets` package is required to load TruthfulQA. Please install it via `pip install datasets`.")
        return []

    try:
        dataset = load_dataset("truthful_qa", "generation", split="validation")
    except Exception as exc:  # pragma: no cover - network/dependency failures
        print(f"Failed to load TruthfulQA: {exc}")
        return []

    samples: List[Dict[str, object]] = []
    for row in dataset:
        question = str(row.get("question") or "").strip()

        correct_answers: Iterable[str] = row.get("correct_answers") or []
        incorrect_answers: Iterable[str] = row.get("incorrect_answers") or []

        for answer_text in correct_answers:
            trace = [
                {"text": f"Question: {question}"},
                {"text": f"Answer: {answer_text}"},
            ]
            samples.append({"trace": trace, "label": 0})
            break

        for answer_text in incorrect_answers:
            trace = [
                {"text": f"Question: {question}"},
                {"text": f"Answer: {answer_text}"},
            ]
            samples.append({"trace": trace, "label": 1})
            break

    return samples


def _load_sample_data(dataset_name: str) -> List[Dict[str, object]]:
    """Provide small, deterministic traces for offline CI and demos."""

    samples: List[Dict[str, object]] = [
        {
            "trace": [
                {"text": "Question: What is the capital of France?"},
                {"text": "Answer: Paris"},
            ],
            "label": 0,
        },
        {
            "trace": [
                {"text": "Question: Compute Fibonacci(6)."},
                {"text": "Answer: Returned 5 after stopping the loop early."},
            ],
            "label": 1,
        },
    ]

    sample_path = Path(__file__).resolve().parent.parent / "tools" / "real_traces" / "sample_gpt4_bad.json"
    if sample_path.exists():
        try:
            trace_steps = json.loads(sample_path.read_text())
            trace_text = " | ".join(str(step.get("text", "")) for step in trace_steps)
            samples.append({"trace": [{"text": trace_text}], "label": 1})
        except Exception as exc:  # pragma: no cover - defensive logging only
            print(f"Warning: failed to load bundled trace {sample_path}: {exc}")

    print(f"Using bundled sample data for {dataset_name} (offline mode).")
    return samples


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=("Benchmark spectral certificates on real datasets, with an offline sample fallback."),
    )
    parser.add_argument(
        "--dataset",
        choices=["halueval", "truthfulqa"],
        default="halueval",
        help="Dataset to evaluate; defaults to HaluEval.",
    )
    parser.add_argument(
        "--use-sample-data",
        action="store_true",
        help="Force usage of bundled sample traces instead of downloading datasets (for CI).",
    )
    args = parser.parse_args(argv)

    dataset_loaders = {
        "halueval": _load_halueval,
        "truthfulqa": _load_truthfulqa,
    }

    samples: List[Dict[str, object]]
    if args.use_sample_data:
        samples = _load_sample_data(args.dataset)
    else:
        samples = dataset_loaders[args.dataset]()
        if not samples:
            print("Falling back to bundled sample data because the real dataset was unavailable.")
            samples = _load_sample_data(args.dataset)

    if not samples:
        print("No samples loaded; exiting.")
        return 1

    results = evaluate_dataset(samples)
    print(f"{args.dataset.capitalize()} Benchmark Results:")
    print_metrics(results)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
