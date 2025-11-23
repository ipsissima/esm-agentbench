from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request

import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentbeats.purple import PurpleAgent

app = Flask(__name__)
logger = logging.getLogger(__name__)


@app.route("/.well-known/agent-card.json")
def agent_card() -> Any:
    card = {
        "name": "tutorial-purple",
        "description": "Baseline debate agent",
        "entrypoint": request.host_url.rstrip("/") + "/run",
    }
    return jsonify(card)


@app.route("/run", methods=["POST"])
def run_agent() -> Any:
    payload: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
    prompt = payload.get("prompt", "demo prompt")
    steps = int(payload.get("max_steps", 5))
    seed = int(payload.get("seed", 0))
    agent = PurpleAgent(seed=seed)
    trace = agent.run(prompt, max_steps=steps)
    return jsonify({"trace": trace})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline purple debater server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9019)
    parser.add_argument("--card-url", default="")
    parser.add_argument("--show-logs", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.show_logs else logging.WARNING)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
