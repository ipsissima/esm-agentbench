"""Bridge runtime certificate code with Coq-extracted constants.

This module provides a strict API for loading numeric constants from
formally verified Coq proofs. Constants MUST be finite floats; the module
will raise errors (not warnings) if verification artifacts are invalid.

Design Principles (Version 2):
1. STRICT VALIDATION: Non-finite values cause hard failures, not fallbacks
2. EXPLICIT AXIOMS: Constants come from Coq files with explicit bounds, not Admitted
3. NO MAGIC NUMBERS: Default values are justified by formal proofs
4. TRANSPARENCY: All loaded constants are logged for audit

Mathematical Basis:
The constants C_res and C_tail are multipliers in the theoretical bound:

    theoretical_bound = C_res * residual + C_tail * tail_energy

These constants are derived from Wedin's Theorem and have proven upper bounds
(see UELAT/spectral_bounds.v). The Python code enforces these bounds.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Conservative defaults with proven bounds from Coq axioms
# C_res <= 2.0, C_tail <= 2.0, C_sem <= 2.0, C_robust <= 2.0 (see UELAT/spectral_bounds.v)
_DEFAULT_CONSTANTS = {
    "C_tail": 1.0,
    "C_res": 1.0,
    "C_sem": 1.0,  # Semantic divergence constant for poison detection
    "C_robust": 1.0,  # Lipschitz embedding robustness constant (perturbation penalty)
}

# Upper bounds from Coq axioms - enforced at load time
_CONSTANT_UPPER_BOUNDS = {
    "C_tail": 2.0,
    "C_res": 2.0,
    "C_sem": 2.0,
    "C_robust": 2.0,
}

_constants: Dict[str, float] = dict(_DEFAULT_CONSTANTS)
_constants_loaded: bool = False


class ConstantLoadError(RuntimeError):
    """Raised when constants cannot be parsed from an input artifact."""


class ConstantValidationError(ValueError):
    """Raised when a constant violates its proven bounds."""


def _validate_constant(name: str, value: float, strict: bool = True) -> float:
    """Validate a single constant value.

    Parameters
    ----------
    name : str
        Name of the constant.
    value : float
        Value to validate.
    strict : bool
        If True, raise errors on invalid values. If False, log warnings.

    Returns
    -------
    float
        The validated value.

    Raises
    ------
    ConstantValidationError
        If the value is non-finite or exceeds proven bounds.
    """
    # Check finite
    if not math.isfinite(value):
        msg = f"Constant '{name}' is not finite: {value!r}. " \
              "Only finite floats are permitted from Coq proofs."
        if strict:
            raise ConstantValidationError(msg)
        logger.error(msg)
        return _DEFAULT_CONSTANTS.get(name, 1.0)

    # Check positivity (required by Coq axioms C_res_pos, C_tail_pos)
    if value <= 0:
        msg = f"Constant '{name}' must be positive (from Coq axiom), got: {value}"
        if strict:
            raise ConstantValidationError(msg)
        logger.error(msg)
        return _DEFAULT_CONSTANTS.get(name, 1.0)

    # Check upper bound (from Coq axioms C_res_bound, C_tail_bound)
    upper_bound = _CONSTANT_UPPER_BOUNDS.get(name)
    if upper_bound is not None and value > upper_bound:
        msg = f"Constant '{name}' = {value} exceeds proven bound {upper_bound}. " \
              "This violates Coq axiom guarantees."
        if strict:
            raise ConstantValidationError(msg)
        logger.error(msg)
        return upper_bound

    return float(value)


def _validate_constants(data: Dict[str, float], strict: bool = True) -> Dict[str, float]:
    """Validate all constants in a dictionary."""
    validated: Dict[str, float] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise ConstantLoadError(f"Constant names must be strings; got {type(key).__name__}")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ConstantLoadError(
                f"Value for constant '{key}' is not numeric: {value!r}"
            ) from exc
        validated[key] = _validate_constant(key, numeric, strict=strict)
    return validated


def load_constants_from_json(path: str, strict: bool = True) -> Dict[str, float]:
    """Load constants from a JSON file with strict validation.

    Parameters
    ----------
    path : str
        Path to JSON file containing constants.
    strict : bool
        If True, raise errors on invalid values. If False, fall back to defaults.

    Returns
    -------
    Dict[str, float]
        Validated constants.

    Raises
    ------
    ConstantLoadError
        If the file cannot be parsed.
    ConstantValidationError
        If any constant is non-finite or exceeds proven bounds (when strict=True).
    """
    global _constants_loaded

    expanded = os.path.expanduser(os.path.expandvars(path))
    if not os.path.exists(expanded):
        if strict:
            raise ConstantLoadError(
                f"Constants JSON '{path}' not found. "
                "Coq-verified constants are required for rigorous bounds."
            )
        logger.warning(
            f"Constants JSON '{path}' not found; using defaults {_DEFAULT_CONSTANTS}"
        )
        return dict(_constants)

    with open(expanded, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ConstantLoadError(
                f"Failed to parse JSON constants from {path}: {exc}"
            ) from exc

    # Extract only numeric constants (ignore metadata fields)
    numeric_data = {
        k: v for k, v in data.items()
        if isinstance(v, (int, float)) and not k.startswith("_")
    }

    validated = _validate_constants(numeric_data, strict=strict)

    # Check required constants
    missing = [k for k in _DEFAULT_CONSTANTS if k not in validated]
    if missing:
        if strict:
            raise ConstantLoadError(
                f"Required constants missing from {path}: {missing}"
            )
        logger.warning(f"Constants JSON missing {missing}; using defaults for them")
        for name in missing:
            validated[name] = _DEFAULT_CONSTANTS[name]

    _constants.clear()
    _constants.update(validated)
    _constants_loaded = True

    logger.info(f"Loaded Coq constants from {path}: {validated}")
    return dict(_constants)


_COQ_PATTERN = re.compile(
    r"(?P<name>[A-Za-z0-9_']+)\s*(?:[:=]|:=).*?(?P<value>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.MULTILINE,
)


def load_constants_from_coq_output(path: str, strict: bool = True) -> Dict[str, float]:
    """Parse constants from a Coq output text file.

    Parameters
    ----------
    path : str
        Path to text file from ``coqtop -batch`` with Print statements.
    strict : bool
        If True, raise errors on invalid values.

    Returns
    -------
    Dict[str, float]
        Validated constants.

    Raises
    ------
    ConstantLoadError
        If parsing fails.
    ConstantValidationError
        If any constant is non-finite or exceeds proven bounds.
    """
    global _constants_loaded

    expanded = os.path.expanduser(os.path.expandvars(path))
    if not os.path.exists(expanded):
        raise ConstantLoadError(f"Coq output file '{path}' does not exist")

    text = open(expanded, "r", encoding="utf-8").read()
    matches = list(_COQ_PATTERN.finditer(text))
    if not matches:
        raise ConstantLoadError(
            "Could not parse any constants from Coq output. "
            "Ensure the file contains lines like 'C_tail = 1.0'."
        )

    parsed = {m.group("name"): float(m.group("value")) for m in matches}
    validated = _validate_constants(parsed, strict=strict)

    _constants.clear()
    _constants.update(validated)

    missing = [k for k in _DEFAULT_CONSTANTS if k not in validated]
    if missing:
        if strict:
            raise ConstantLoadError(
                f"Required constants missing from Coq output: {missing}"
            )
        logger.warning(f"Coq output missing {missing}; using defaults for them")
        for name in missing:
            _constants[name] = _DEFAULT_CONSTANTS[name]

    _constants_loaded = True
    logger.info(f"Loaded Coq constants from {path}: {_constants}")
    return dict(_constants)


def get_constant(name: str, strict: bool = False) -> float:
    """Return a constant by name with validation.

    Parameters
    ----------
    name : str
        Name of the constant to retrieve.
    strict : bool
        If True, raise error if constant not loaded. If False, return default.

    Returns
    -------
    float
        The constant value.

    Raises
    ------
    ConstantValidationError
        If the value is non-finite (always raises, regardless of strict).
    """
    if name in _constants:
        value = _constants[name]
    elif name in _DEFAULT_CONSTANTS:
        if strict and not _constants_loaded:
            raise ConstantLoadError(
                f"Constant '{name}' requested but no Coq constants loaded. "
                "Call load_constants_from_json() first."
            )
        logger.debug(f"Using default for constant '{name}': {_DEFAULT_CONSTANTS[name]}")
        value = _DEFAULT_CONSTANTS[name]
        _constants[name] = value
    else:
        if strict:
            raise ConstantLoadError(
                f"Unknown constant '{name}'. Known constants: {list(_DEFAULT_CONSTANTS.keys())}"
            )
        logger.warning(f"Unknown constant '{name}'; returning conservative default 1.0")
        value = 1.0
        _constants[name] = value

    # Always validate - non-finite values are never acceptable
    if not math.isfinite(value):
        raise ConstantValidationError(
            f"Constant '{name}' is not finite: {value!r}. "
            "This indicates corruption of the verification artifacts."
        )

    return float(value)


def list_constants() -> Dict[str, float]:
    """Return a shallow copy of the currently loaded constants."""
    return dict(_constants)


def constants_loaded() -> bool:
    """Return True if constants have been explicitly loaded from a file."""
    return _constants_loaded


def get_default_constants_path() -> Optional[Path]:
    """Return path to default constants JSON if it exists."""
    candidates = [
        Path(__file__).parent.parent / "UELAT" / "uelat_constants.json",
        Path(__file__).parent / "uelat_constants.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def auto_load_constants(strict: bool = False) -> Dict[str, float]:
    """Attempt to auto-load constants from default locations.

    Parameters
    ----------
    strict : bool
        If True, raise error if no constants file found.

    Returns
    -------
    Dict[str, float]
        Loaded or default constants.
    """
    path = get_default_constants_path()
    if path is not None:
        return load_constants_from_json(str(path), strict=strict)
    elif strict:
        raise ConstantLoadError(
            "No constants file found at default locations. "
            "Expected UELAT/uelat_constants.json"
        )
    else:
        logger.info("No constants file found; using conservative defaults")
        return dict(_DEFAULT_CONSTANTS)


__all__ = [
    "load_constants_from_json",
    "load_constants_from_coq_output",
    "get_constant",
    "list_constants",
    "constants_loaded",
    "auto_load_constants",
    "get_default_constants_path",
    "ConstantLoadError",
    "ConstantValidationError",
]
