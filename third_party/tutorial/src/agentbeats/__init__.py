"""Lightweight AgentBeats tutorial compatibility layer.

This module provides a minimal harness so integration tests can run
offline without the full upstream tutorial package. The real tutorial
assets can be imported via ``tools/import_tutorial.py`` which populates
``third_party/tutorial`` from the provided archive.
"""
from .base import Assessment, GreenExecutorBase
from .artifacts import send_artifact, send_task_update

__all__ = ["Assessment", "GreenExecutorBase", "send_artifact", "send_task_update"]
