"""Offline driver to run the mini SWE-bench style demo episodes."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.abspath("demo_swe/refactor_sources"))
sys.path.insert(0, str(REPO_ROOT / "demo_swe" / "refactor_sources"))

EPISODE_DIR = REPO_ROOT / "demo_swe" / "episodes"
TRACE_DIR = REPO_ROOT / "demo_traces"
REPORT_PATH = REPO_ROOT / "demo_swe" / "report.json"
ASSESS_ENDPOINT = os.environ.get("ASSESS_ENDPOINT", "http://localhost:8080/run_episode")

TRACE_DIR.mkdir(exist_ok=True)
REPORT_PATH.parent.mkdir(exist_ok=True)


def _read_episode(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Episode {path} must be a JSON object")
    for key in ("id", "prompt", "tests"):
        if key not in data:
            raise ValueError(f"Episode {path} missing required key '{key}'")
    return data


def _post_json(url: str, payload: Dict[str, Any], retries: int = 2, delay: float = 1.0) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    last_err: Exception | None = None
    timeout_sec = float(os.environ.get("DEMO_HTTP_TIMEOUT", "20"))
    for attempt in range(retries):
        try:
            req = Request(url=url, data=body, headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(req, timeout=timeout_sec) as resp:
                text = resp.read().decode("utf-8")
                return json.loads(text)
        except HTTPError as exc:
            try:
                text = exc.read().decode("utf-8")
            except Exception:
                text = str(exc)
            if exc.code == 400:
                raise RuntimeError(f"HTTP 400 from assessor: {text}")
            last_err = exc
            time.sleep(delay)
        except (URLError, TimeoutError, ValueError) as exc:
            last_err = exc
            time.sleep(delay)
    raise RuntimeError(f"Failed to POST after {retries} attempts: {last_err}")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON atomically to avoid partially written demo outputs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as tmp:
            json.dump(payload, tmp, ensure_ascii=False, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def run_demo() -> Dict[str, Any]:
    episodes = sorted(EPISODE_DIR.glob("ep*.json"))
    max_eps = int(os.environ.get("DEMO_MAX_EPISODES", "3"))
    if max_eps > 0:
        episodes = episodes[:max_eps]
    summaries: List[Dict[str, Any]] = []

    for ep_path in episodes:
        episode = _read_episode(ep_path)
        try:
            response = _post_json(ASSESS_ENDPOINT, episode)
        except RuntimeError as exc:
            print(f"Episode {episode.get('id')} failed: {exc}", file=sys.stderr)
            raise
        except Exception as exc:
            # Preserve diagnostics so demo tests can inspect errors without crashing.
            response = {
                "episode_id": episode.get("id"),
                "task_success": False,
                "task_score": 0.0,
                "spectral_metrics": {},
                "trace_path": None,
                "error": str(exc),
            }

        episode_id = response.get("episode_id", episode["id"])
        spectral = response.get("spectral_metrics", {})
        certificate_path = TRACE_DIR / f"{episode_id}_certificate.json"

        certificate = {
            "episode": episode.get("id"),
            "episode_id": episode_id,
            "spectral_metrics": spectral,
            "task_success": bool(response.get("task_success", False)),
            "task_score": float(response.get("task_score", 0.0)),
            "trace_path": response.get("trace_path"),
        }
        _atomic_write_json(certificate_path, certificate)

        summaries.append(
            {
                "episode": episode.get("id"),
                "episode_id": episode_id,
                "task_success": bool(response.get("task_success", False)),
                "task_score": float(response.get("task_score", 0.0)),
                "trace_path": response.get("trace_path"),
                "certificate_path": str(certificate_path.relative_to(REPO_ROOT)),
                "spectral_metrics": spectral,
                "error": response.get("error"),
            }
        )

    residuals = [float(s.get("spectral_metrics", {}).get("residual", 0.0)) for s in summaries]
    bounds = [float(s.get("spectral_metrics", {}).get("theoretical_bound", 0.0)) for s in summaries]
    task_success_rate = float(np.mean([s.get("task_success", False) for s in summaries])) if summaries else 0.0

    report = {
        "episodes": summaries,
        "aggregate": {
            "task_success_rate": task_success_rate,
            "mean_residual": float(np.mean(residuals)) if residuals else 0.0,
            "mean_theoretical_bound": float(np.mean(bounds)) if bounds else 0.0,
        },
        "notes": "Deterministic offline SWE demo",
    }

    _atomic_write_json(REPORT_PATH, report)

    return report


if __name__ == "__main__":  # pragma: no cover
    output = run_demo()
    print(json.dumps(output, indent=2))
