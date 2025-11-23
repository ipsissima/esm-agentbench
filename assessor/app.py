"""Minimal Flask application exposing agent card and episode runner."""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request

# Lazy TOML loader compatible with Python 3.9+
try:  # Python 3.11+
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for <=3.10
    import tomli as tomllib  # type: ignore

from assessor import kickoff
from certificates.make_certificate import compute_certificate

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_CARD_PATH = REPO_ROOT / "agent_card.toml"
TRACE_DIR = REPO_ROOT / "demo_traces"
TRACE_DIR.mkdir(exist_ok=True)


def _load_agent_card() -> Dict[str, Any]:
    """Load the agent_card.toml from repository root."""
    if not AGENT_CARD_PATH.exists():
        raise FileNotFoundError("agent_card.toml is missing at repository root")
    try:
        with AGENT_CARD_PATH.open("rb") as f:
            data = tomllib.load(f)
        return data
    except Exception as exc:
        logging.error("agent_card.toml is malformed: %s", exc)
        return {
            "name": "ESM Spectral Assessor",
            "entrypoint": "http://localhost:8080/.well-known/agent-card.json",
            "description": "Fallback card due to parse error",
        }


@app.route("/.well-known/agent-card.json", methods=["GET"])
def agent_card() -> Any:
    """Return the agent card as JSON or a helpful 404."""
    try:
        card = _load_agent_card()
        return jsonify(card), 200
    except FileNotFoundError as exc:
        logging.error("Agent card missing: %s", exc)
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:  # pragma: no cover - defensive
        logging.exception("Failed to load agent card")
        return jsonify({"error": "failed to load agent card", "detail": str(exc)}), 500


@app.route("/run_episode", methods=["POST"])
def run_episode() -> Any:
    """Execute a demo episode and return task and spectral metrics."""
    try:
        payload = request.get_json(force=True, silent=False) or {}
        if not isinstance(payload, dict):
            raise ValueError("Expected JSON object in request body")
        task_spec = payload
        if "prompt" not in task_spec:
            raise ValueError("task_spec must include 'prompt'")

        result = kickoff.run_episode(task_spec)
        episode_id = result["episode_id"]
        trace = result["trace"]
        trace_path = Path(result.get("trace_path", TRACE_DIR / f"{episode_id}.json"))

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
            "trace_path": str(trace_path.relative_to(REPO_ROOT)),
            "early_warning_step": result.get("early_warning_step"),
        }
        return jsonify(response), 200
    except Exception as exc:  # pragma: no cover - safety net
        logging.exception("/run_episode failed")
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":  # pragma: no cover
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
