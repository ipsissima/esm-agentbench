"""Generate Good vs Bad Chain-of-Thought traces and visualize residuals."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

from assessor import kickoff

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACE_DIR = REPO_ROOT / "demo_traces"
REPORT_DIR = REPO_ROOT / "demo_swe"
REPORT_PATH = REPORT_DIR / "report.json"
FIG_PATH = REPORT_DIR / "fig_hallucination.png"

TRACE_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


FIB_TESTS = [
    {"name": "fib_0", "script": "assert fibonacci(0) == 0"},
    {"name": "fib_1", "script": "assert fibonacci(1) == 1"},
    {"name": "fib_5", "script": "assert fibonacci(5) == 5"},
]


def _cumulative_residuals(trace: List[Dict[str, Any]]) -> List[float]:
    total = 0.0
    series: List[float] = []
    for entry in trace:
        resid = entry.get("residual")
        if resid is not None:
            total += float(resid)
        series.append(total)
    return series


def _run_episode(prompt: str, name: str) -> Dict[str, Any]:
    spec = {"prompt": prompt, "tests": FIB_TESTS, "trace_dir": TRACE_DIR}
    result = kickoff.run_episode(spec, max_steps=12)
    result["name"] = name
    return result


def main() -> None:
    episodes = [
        ("Good", "Fibonacci Chain-of-Thought with correct steps"),
        ("Bad", "Bad CoT about cakes while solving Fibonacci"),
    ]

    results: List[Dict[str, Any]] = []
    for name, prompt in episodes:
        results.append(_run_episode(prompt, name))

    plt.figure(figsize=(8, 4))
    for res in results:
        cum_resid = _cumulative_residuals(res["trace"])
        plt.plot(cum_resid, label=f"{res['name']} (success={res['task_success']})")
    plt.xlabel("Step")
    plt.ylabel("Cumulative residual")
    plt.title("Good vs Bad CoT residual trajectories")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_PATH)

    report = {
        "episodes": [
            {
                "name": res.get("name"),
                "episode_id": res.get("episode_id"),
                "task_success": res.get("task_success"),
                "task_score": res.get("task_score"),
                "trace_path": str(Path(res.get("trace_path", "")).relative_to(REPO_ROOT)),
                "early_warning_step": res.get("early_warning_step"),
            }
            for res in results
        ],
        "figure": str(FIG_PATH.relative_to(REPO_ROOT)),
    }

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
