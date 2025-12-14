"""Runnable server exposing the ESM green executor."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from esmassessor.base import Assessment
from esmassessor.green_executor import EsmGreenExecutor

# Import for /run_episode endpoint
from assessor import kickoff
from certificates.make_certificate import compute_certificate

app = Flask(__name__)
LOGGER = logging.getLogger(__name__)
EXECUTOR = EsmGreenExecutor()
TRACE_DIR = PROJECT_ROOT / "demo_traces"


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


@app.route("/run_episode", methods=["POST"])
def run_episode() -> Any:
    """Execute a demo episode and return task and spectral metrics."""
    try:
        payload = request.get_json(force=True, silent=False) or {}
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected JSON object in request body"}), 400
        task_spec = payload
        if "prompt" not in task_spec:
            return jsonify({"error": "task_spec must include 'prompt'"}), 400

        result = kickoff.run_episode(task_spec)
        episode_id = result["episode_id"]
        trace = result["trace"]
        trace_path = Path(result.get("trace_path", TRACE_DIR / f"{episode_id}.json"))

        # Ensure trace directory exists
        TRACE_DIR.mkdir(parents=True, exist_ok=True)

        if not trace_path.exists():
            trace_path = TRACE_DIR / f"{episode_id}.json"
            with trace_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        certificate = result.get("certificate")
        if not certificate:
            embeddings = kickoff.embed_trace_steps(trace)
            certificate = compute_certificate(embeddings)
        certificate["confidence"] = max(0.0, 1.0 - certificate.get("residual", 1.0))

        response = {
            "episode_id": episode_id,
            "task_success": bool(result.get("task_success", False)),
            "task_score": float(result.get("task_score", 0.0)),
            "spectral_metrics": certificate,
            "trace_path": str(trace_path.relative_to(PROJECT_ROOT)),
            "early_warning_step": result.get("early_warning_step"),
        }
        return jsonify(response), 200
    except Exception as exc:
        LOGGER.exception("run_episode failed")
        return jsonify({"error": str(exc)}), 500


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ESM green server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--serve-only", action="store_true", help="serve HTTP without running demo")
    parser.add_argument("--show-logs", action="store_true", help="enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.show_logs else logging.WARNING)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
