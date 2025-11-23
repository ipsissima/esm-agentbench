import json
import os
import subprocess
import socket
import threading
import time
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from werkzeug.serving import make_server

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from assessor.app import app
def _allocate_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


PORT = _allocate_port()
DEMO_DRIVER = REPO_ROOT / "tools" / "run_demo_swe.py"
REPORT_PATH = REPO_ROOT / "demo_swe" / "report.json"


def _wait_for_app(timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(f"http://localhost:{PORT}/.well-known/agent-card.json", timeout=2):
                return
        except HTTPError:
            return
        except URLError:
            time.sleep(0.5)
    raise RuntimeError("assessor app did not start in time")


def _start_app():
    server = make_server("127.0.0.1", PORT, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_demo_driver_runs_offline():
    server = _start_app()
    try:
        _wait_for_app()
        try:
            run = subprocess.run(
                ["python", str(DEMO_DRIVER)],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "ASSESS_ENDPOINT": f"http://localhost:{PORT}/run_episode"},
            )
        except subprocess.CalledProcessError as exc:
            print(exc.stdout)
            print(exc.stderr)
            raise
        assert REPORT_PATH.exists(), run.stdout + run.stderr
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        assert "episodes" in report and report["episodes"], "Report missing episodes"
        for ep in report["episodes"]:
            cert_fields = ep.get("spectral_metrics", {})
            for key in [
                "pca_explained",
                "max_eig",
                "spectral_gap",
                "residual",
                "pca_tail_estimate",
                "theoretical_bound",
            ]:
                assert key in cert_fields
            cert_path = Path(ep["certificate_path"])
            if not cert_path.is_absolute():
                cert_path = REPO_ROOT / cert_path
            assert cert_path.exists()
            cert_json = json.loads(cert_path.read_text(encoding="utf-8"))
            assert "spectral_metrics" in cert_json
    finally:
        server.shutdown()
