"""Spectral analysis tools for agent trajectory certification.

This module provides mathematical tools for analyzing agent behavior traces
using spectral methods. The core approach is based on the discrete Telegrapher
equation with Koopman-inspired system identification.

Key components:
- mass_model: Second-difference penalty regression for drift detection
"""
from __future__ import annotations

from .mass_model import (
    fit_mass_model,
    MassModelResult,
    compute_mass_residual,
    synthetic_rabbit_trace,
)

__all__ = [
    "fit_mass_model",
    "MassModelResult",
    "compute_mass_residual",
    "synthetic_rabbit_trace",
]
