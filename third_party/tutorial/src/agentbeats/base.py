from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, MutableMapping, Optional

logger = logging.getLogger(__name__)


@dataclass
class Assessment:
    """Container for a single assessment episode."""

    assessment_id: str
    prompt: str
    participants: List[str]
    max_steps: int = 8
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


class GreenExecutorBase:
    """Minimal base class mirrored from the AgentBeats tutorial API."""

    def __init__(self, assessment_timeout: float = 120.0) -> None:
        self.assessment_timeout = assessment_timeout

    def assess(self, assessment: Assessment, traces: Mapping[str, List[Mapping[str, Any]]]) -> Dict[str, Any]:
        raise NotImplementedError

    # Hooks used by integration harness
    def on_start(self, assessment: Assessment) -> None:  # pragma: no cover - hooks are optional
        logger.info("starting assessment %s", assessment.assessment_id)

    def on_complete(self, assessment: Assessment, result: Mapping[str, Any]) -> None:  # pragma: no cover
        logger.info("completed assessment %s", assessment.assessment_id)
