from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def send_artifact(artifact: Mapping[str, Any], name: str = "artifact.json", directory: Optional[Path] = None) -> Path:
    """Persist artifact locally and log the event.

    The real tutorial client would POST artifacts back to the orchestrator.
    For offline integration we simply serialize to disk under ``demo_traces``
    (or the provided ``directory``).
    """

    base_dir = Path(directory) if directory else Path("demo_traces")
    _ensure_dir(base_dir / name)
    out_path = base_dir / name
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    logger.info("sent artifact %s", out_path)
    return out_path


def send_task_update(payload: Mapping[str, Any]) -> None:
    """Log task updates for traceability."""

    logger.info("task update: %s", payload)
