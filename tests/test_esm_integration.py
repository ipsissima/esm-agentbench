from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def test_green_and_purple_start():
    env = os.environ.copy()
    env["ESM_FORCE_TFIDF"] = "1"
    green = subprocess.Popen([
        "python",
        "-m",
        "esmassessor.green_server",
        "--host",
        "127.0.0.1",
        "--port",
        "8099",
        "--serve-only",
    ], env=env)
    purple = subprocess.Popen([
        "python",
        "scenarios/debate/debater.py",
        "--host",
        "127.0.0.1",
        "--port",
        "9020",
    ])
    time.sleep(1.5)
    purple.terminate()
    green.terminate()
    purple.wait(timeout=5)
    green.wait(timeout=5)


def test_agentbeats_run(tmp_path):
    env = os.environ.copy()
    env["ESM_FORCE_TFIDF"] = "1"
    shutil.rmtree("demo_traces", ignore_errors=True)
    subprocess.check_call([
        "./agentbeats-run",
        "scenarios/esm/scenario.toml",
        "--show-logs",
    ], env=env)
    certs = sorted(Path("demo_traces").glob("*_certificate.json"))
    assert certs, "certificate files missing"
    data = json.loads(certs[0].read_text())
    metrics = data.get("spectral_metrics", {})
    assert "residual" in metrics or "theoretical_bound" in metrics
