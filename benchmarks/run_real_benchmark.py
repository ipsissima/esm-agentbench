"""Run spectral certificate benchmarks on real datasets.

This script loads public hallucination datasets from Hugging Face, embeds each
trace with :func:`assessor.kickoff.embed_trace_steps`, and reports the
AUC-ROC of the spectral certificate's ``theoretical_bound`` against ground
truth labels.
"""
from __future__ import annotations

import argparse
import sys
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark spectral certificates on real datasets.")
    parser.add_argument("--dataset", choices=["halueval", "truthfulqa"], required=True, help="Dataset to evaluate.")
    args = parser.parse_args(argv)

    if args.dataset == "halueval":
        samples = _load_halueval()
    else:
        samples = _load_truthfulqa()

    if not samples:
        print("No samples loaded; exiting.")
        return 1

    results = evaluate_dataset(samples)
    print(f"{args.dataset.capitalize()} Benchmark Results:")
    print_metrics(results)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
