"""Inference port definitions."""
from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, runtime_checkable


@runtime_checkable
class InferencePort(Protocol):
    """Port for model inference implementations."""

    def run_inference(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        """Run inference and return a structured response."""
