#!/usr/bin/env python
"""Ingest raw LLM transcripts into structured trace JSON files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parent.parent))

from assessor.kickoff import _utc_iso, _short_context


def transcript_to_trace(text: str) -> List[dict]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    trace: List[dict] = []
    context_accum = ""
    for idx, line in enumerate(lines, start=1):
        context_accum = _short_context(context_accum + "\n" + line)
        trace.append(
            {
                "step": idx,
                "role": "agent",
                "type": "thought",
                "text": line,
                "timestamp": _utc_iso(),
                "context": context_accum,
            }
        )
    return trace


def write_trace(trace: List[dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Convert raw transcripts to trace JSON files")
    parser.add_argument("input_dir", type=Path, help="Directory of raw .txt transcripts")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo_traces"),
        help="Where to store converted traces.",
    )
    args = parser.parse_args()

    for txt_path in args.input_dir.glob("*.txt"):
        trace = transcript_to_trace(txt_path.read_text(encoding="utf-8"))
        out_path = args.output_dir / f"{txt_path.stem}.json"
        write_trace(trace, out_path)
        print("Wrote", out_path)

    # ship a couple of curated samples for offline use
    sample_dir = Path(__file__).resolve().parent / "real_traces"
    sample_dir.mkdir(exist_ok=True)
    write_trace(sample_good_trace(), sample_dir / "sample_gpt4_good.json")
    write_trace(sample_bad_trace(), sample_dir / "sample_gpt4_bad.json")


def sample_good_trace() -> List[dict]:
    return [
        {"step": 0, "role": "agent", "type": "thought", "text": "Plan: compute Fibonacci", "timestamp": _utc_iso()},
        {"step": 1, "role": "agent", "type": "thought", "text": "Step 1: handle n<2", "timestamp": _utc_iso()},
        {"step": 2, "role": "agent", "type": "thought", "text": "Step 2: iterate", "timestamp": _utc_iso()},
        {"step": 3, "role": "agent", "type": "code", "text": "```python\n" "def fib(n):\n    a,b=0,1\n    for _ in range(n):\n        a,b=b,a+b\n    return a\n```", "timestamp": _utc_iso()},
        {"step": 4, "role": "agent", "type": "thought", "text": "Return result", "timestamp": _utc_iso()},
    ]


def sample_bad_trace() -> List[dict]:
    return [
        {"step": 0, "role": "agent", "type": "thought", "text": "Plan: discuss Fibonacci", "timestamp": _utc_iso()},
        {"step": 1, "role": "agent", "type": "thought", "text": "Step 1: describe rabbits", "timestamp": _utc_iso()},
        {"step": 2, "role": "agent", "type": "thought", "text": "Step 2: tangent about cake", "timestamp": _utc_iso()},
        {"step": 3, "role": "agent", "type": "code", "text": "```python\nprint('not actually computing fib')\n```", "timestamp": _utc_iso()},
        {"step": 4, "role": "agent", "type": "thought", "text": "Claims solved", "timestamp": _utc_iso()},
    ]


if __name__ == "__main__":
    main()
