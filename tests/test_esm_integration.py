from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

from esmassessor.base import Assessment
from esmassessor.green_executor import EsmGreenExecutor


def _wait_for_health(port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("green server did not start in time")


def test_green_server_healthcheck():
    env = os.environ.copy()
    env["ESM_FORCE_TFIDF"] = "1"
    port = 8099
    proc = subprocess.Popen(
        [
            "python",
            "-m",
            "esmassessor.green_server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--serve-only",
        ],
        env=env,
    )
    try:
        _wait_for_health(port)
        with urlopen(f"http://127.0.0.1:{port}/.well-known/agent-card.json", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        assert data.get("name") == "esm-green"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_green_executor_produces_certificate(tmp_path):
    shutil.rmtree("demo_traces", ignore_errors=True)

    assessment = Assessment(assessment_id="unit", prompt="simple trace", participants=["green"], max_steps=3)
    traces = {
        "green": [
            {"participant": "green", "step": i, "text": f"step {i}", "confidence": 0.8} for i in range(3)
        ]
    }

    executor = EsmGreenExecutor()
    result = executor.assess(assessment, traces)
    cert = result.get("green", {})
    metrics = cert.get("spectral_metrics", {})
    assert "residual" in metrics
    expected_path = Path("demo_traces") / "unit_green_certificate.json"
    assert expected_path.exists()
