"""Filesystem trace storage adapter."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from ports.storage import TraceStoragePort


class FilesystemTraceStorage(TraceStoragePort):
    """Store traces as JSON files on disk."""

    def __init__(self, root_dir: Path) -> None:
        self._root_dir = Path(root_dir)
        self._root_dir.mkdir(parents=True, exist_ok=True)

    def save_trace(
        self,
        metadata: Mapping[str, Any],
        steps: Sequence[Mapping[str, Any]],
        outcome: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        payload: Dict[str, Any] = {
            "metadata": dict(metadata),
            "steps": [dict(step) for step in steps],
            "outcome": dict(outcome) if outcome else None,
            "saved_at": datetime.utcnow().isoformat() + "Z",
        }

        trace_id = payload["metadata"].get("trace_id", "trace")
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"{trace_id}_{timestamp}.json"
        trace_path = self._root_dir / filename
        trace_path.write_text(json.dumps(payload, indent=2))
        return trace_path

    def load_trace(self, trace_file: Path) -> Dict[str, Any]:
        return json.loads(Path(trace_file).read_text())
