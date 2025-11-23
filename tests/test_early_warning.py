import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from assessor.kickoff import run_episode  # noqa: E402


def test_early_warning_triggers_for_bad_only():
    good_spec = {
        "prompt": "Compute fibonacci numbers with coherent reasoning.",
        "tests": [
            {"name": "fib", "script": "assert fibonacci(5) == 5"}
        ],
        "residual_threshold": 2.0,
        "max_steps": 8,
    }

    bad_spec = {
        "prompt": "Compute fibonacci numbers but induce bad cot hallucination.",
        "tests": [
            {"name": "fib", "script": "assert fibonacci(4) == 3"}
        ],
        "residual_threshold": 0.05,
        "max_steps": 8,
        "step_prompt_template": (
            "{previous_context}\n\nStep {step_num}/{max_steps}: Continue reasoning about fibonacci with bad cot injection."
        ),
    }

    good = run_episode(good_spec)
    bad = run_episode(bad_spec)

    assert good["early_warning_step"] is None
    assert bad["early_warning_step"] is not None
    assert bad["early_warning_step"] > 0
