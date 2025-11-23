"""Runnable server exposing the ESM green executor."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_SRC = PROJECT_ROOT / "third_party" / "tutorial" / "src"
if TUTORIAL_SRC.exists() and str(TUTORIAL_SRC) not in sys.path:
    sys.path.insert(0, str(TUTORIAL_SRC))

from agentbeats import Assessment
from agentbeats.runner import run_debate
from esmassessor.green_executor import EsmGreenExecutor

app = Flask(__name__)
LOGGER = logging.getLogger(__name__)
EXECUTOR = EsmGreenExecutor()


@app.route("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.route("/.well-known/agent-card.json")
def agent_card() -> Any:
    return jsonify({
        "name": "esm-green",
        "description": "ESM spectral certificate assessor",
        "entrypoint": "/assess",
    })


@app.route("/assess", methods=["POST"])
def assess() -> Any:
    payload: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
    assessment_id = payload.get("assessment_id", "esm_demo")
    prompt = payload.get("prompt", "Debate the merits of Koopman analysis")
    participants = list(payload.get("participants", payload.get("traces", {}).keys()))
    traces = payload.get("traces", {})
    assessment = Assessment(assessment_id=assessment_id, prompt=prompt, participants=list(participants))
    result = EXECUTOR.assess(assessment, traces)
    return jsonify(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ESM green server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--serve-only", action="store_true", help="serve HTTP without running demo")
    parser.add_argument("--show-logs", action="store_true", help="enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.show_logs else logging.WARNING)
    if not args.serve_only:
        # Run a tiny demo debate to warm up embeddings and produce artifacts
        scenario = {
            "assessment_id": "esm_demo",
            "prompt": "quick warm-up",
            "max_steps": 3,
            "participants": ["green", "purple"],
        }
        try:
            run_debate(scenario, show_logs=args.show_logs)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("warm-up debate failed: %s", exc)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
