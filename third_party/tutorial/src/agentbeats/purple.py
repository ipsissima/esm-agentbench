from __future__ import annotations

import logging
import random
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class PurpleAgent:
    """Baseline purple agent producing deterministic debate steps."""

    def __init__(self, name: str = "purple", seed: int = 0) -> None:
        self.name = name
        self.seed = seed

    def run(self, prompt: str, max_steps: int = 6) -> List[Dict[str, Any]]:
        rng = random.Random(self.seed)
        trace: List[Dict[str, Any]] = []
        for step_idx in range(max_steps):
            thought = f"{self.name} defends position {step_idx} on {prompt}"
            confidence = 0.55 + 0.05 * rng.random()
            trace.append({
                "participant": self.name,
                "step": step_idx,
                "text": thought,
                "confidence": round(confidence, 3),
            })
        return trace

    def run_negative(self, prompt: str, max_steps: int = 4) -> List[Dict[str, Any]]:
        rng = random.Random(self.seed + 42)
        trace: List[Dict[str, Any]] = []
        for step_idx in range(max_steps):
            thought = f"{self.name} challenges claim {step_idx}: {prompt}"
            confidence = 0.35 + 0.03 * rng.random()
            trace.append({
                "participant": self.name,
                "step": step_idx,
                "text": thought,
                "confidence": round(confidence, 3),
            })
        return trace
