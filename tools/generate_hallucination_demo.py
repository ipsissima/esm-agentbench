"""Generate good vs bad CoT runs, residual plot, and leaderboard."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from assessor.kickoff import run_episode


def _fib_tests() -> List[Dict[str, str]]:
    return [
        {
            "name": "fib",
            "script": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(7) == 13",
        }
    ]


def _good_spec() -> Dict:
    return {
        "prompt": "Implement fibonacci with iterative approach for reliable CoT.",
        "tests": _fib_tests(),
        "max_steps": 12,
        "residual_threshold": 0.2,
    }


def _bad_spec() -> Dict:
    return {
        "prompt": "Implement fibonacci but sneak in a bad CoT hallucination.",
        "tests": _fib_tests(),
        "max_steps": 8,
        "residual_threshold": 0.05,
        "step_prompt_template": (
            "{previous_context}\n\nStep {step_num}/{max_steps}: Continue reasoning about fibonacci."
            " Mention 'bad cot' to trigger drift."
        ),
    }


def _cumulative_residuals(residuals: List[float | None]) -> np.ndarray:
    clean = [r if r is not None else 0.0 for r in residuals]
    return np.cumsum(clean)


def _write_leaderboard(report_path: Path) -> None:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    rows = []
    for label, record in data.items():
        coherence = 1.0 - float(record.get("theoretical_bound", 1.0))
        coherence = max(0.0, min(1.0, coherence))
        rows.append(
            f"<tr><td>{label}</td><td>{record.get('task_success')}</td>"
            f"<td>{coherence:.3f}</td><td>{record.get('early_warning_step')}</td></tr>"
        )
    table = "\n".join(rows)
    html = f"""
<!doctype html>
<html lang='en'>
<head><meta charset='utf-8'><title>CoT Coherence Leaderboard</title></head>
<body>
<h2>Coherence Leaderboard (offline demo)</h2>
<table border='1' cellpadding='4' cellspacing='0'>
  <tr><th>Agent</th><th>Success</th><th>Coherence</th><th>Early Warnings</th></tr>
  {table}
</table>
<p>Coherence is 1 - theoretical_bound (clipped to [0,1]) for easy visual ranking.</p>
</body>
</html>
"""
    report_path.with_suffix(".html").write_text(html, encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    demo_dir = base_dir / "demo_swe"
    demo_dir.mkdir(exist_ok=True)

    good = run_episode(_good_spec())
    bad = run_episode(_bad_spec())

    series = {
        "Good": _cumulative_residuals(good["per_step_residuals"]),
        "Bad": _cumulative_residuals(bad["per_step_residuals"]),
    }

    plt.figure(figsize=(6, 4))
    for label, values in series.items():
        plt.plot(values, label=label)
    plt.title("Cumulative Residual Drift: Good vs Bad CoT")
    plt.xlabel("Step")
    plt.ylabel("Cumulative residual")
    plt.legend()
    plt.tight_layout()
    fig_path = demo_dir / "fig_hallucination.png"
    plt.savefig(fig_path)
    plt.close()

    report = {
        "good": {
            "task_success": good["task_success"],
            "early_warning_step": good["early_warning_step"],
            "theoretical_bound": good["certificate"].get("theoretical_bound"),
        },
        "bad": {
            "task_success": bad["task_success"],
            "early_warning_step": bad["early_warning_step"],
            "theoretical_bound": bad["certificate"].get("theoretical_bound"),
        },
    }
    report_path = demo_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_leaderboard(report_path)

    print(f"Wrote demo report to {report_path}")
    print(f"Saved hallucination figure to {fig_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
