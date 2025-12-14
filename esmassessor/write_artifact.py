from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping, Optional

from esmassessor.artifacts import send_artifact

LOGGER = logging.getLogger(__name__)


def write_certificate_artifact(data: Mapping[str, object], path: Path, also_send: bool = True) -> Path:
    """Write an artifact to disk and optionally forward via tutorial helper."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    LOGGER.info("wrote artifact %s", path)
    if also_send:
        send_artifact(data, name=path.name, directory=path.parent)
    return path


__all__ = ["write_certificate_artifact"]
