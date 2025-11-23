from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import tomli

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentbeats.base import Assessment
from agentbeats.artifacts import send_artifact, send_task_update
from agentbeats.purple import PurpleAgent

LOGGER = logging.getLogger(__name__)


def _load_scenario(path: Path) -> Mapping[str, Any]:
    data = tomli.loads(path.read_text(encoding="utf-8"))
    return data


def _synthetic_trace(participant: str, prompt: str, max_steps: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    trace: List[Dict[str, Any]] = []
    for idx in range(max_steps):
        trace.append(
            {
                "participant": participant,
                "step": idx,
                "text": f"{participant} reasoning {idx} about {prompt} :: {rng.random():.3f}",
                "evidence": rng.random(),
            }
        )
    return trace


def run_debate(scenario: Mapping[str, Any], show_logs: bool = False) -> Dict[str, Any]:
    from esmassessor.green_executor import EsmGreenExecutor

    assessment_id = scenario.get("assessment_id", "esm_demo")
    prompt = scenario.get("prompt", "Debate the benefits of spectral certificates.")
    max_steps = int(scenario.get("max_steps", 6))
    seed = int(scenario.get("seed", 0))
    participants = list(scenario.get("participants", ["green", "purple"]))
    assessment = Assessment(assessment_id=assessment_id, prompt=prompt, participants=participants, max_steps=max_steps)

    purple = PurpleAgent(name="purple", seed=seed)

    traces: Dict[str, List[Dict[str, Any]]] = {}
    for name in participants:
        if name == "purple":
            traces[name] = purple.run(prompt, max_steps=max_steps)
        else:
            traces[name] = _synthetic_trace(name, prompt, max_steps=max_steps, seed=seed + len(name))

    executor = EsmGreenExecutor()
    if show_logs:
        LOGGER.info("starting debate scenario %s", assessment_id)
    result = executor.assess(assessment, traces)
    overall_artifact = {
        "assessment_id": assessment_id,
        "prompt": prompt,
        "participants": participants,
        "summary": result,
    }
    send_artifact(overall_artifact, name="certificate.json")
    return overall_artifact


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run AgentBeats tutorial scenario")
    parser.add_argument("scenario", type=Path, nargs="?", default=Path("scenarios/esm/scenario.toml"))
    parser.add_argument("--show-logs", action="store_true", help="enable verbose logging")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.show_logs else logging.WARNING)
    if not args.scenario.exists():
        raise SystemExit(f"Scenario file not found: {args.scenario}")
    scenario = _load_scenario(args.scenario)
    payload = run_debate(scenario, show_logs=args.show_logs)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
