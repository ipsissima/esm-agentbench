"""Bridge runtime certificate code with Coq-extracted constants.

This module provides a minimal API for loading numeric constants that appear in
Coq lemmas justifying the finite-rank bound. Constants can be consumed either
from a JSON artifact (preferred) or from raw Coq output captured from
``coqtop -batch``. The JSON schema is ``{ "NAME": <number>, ... }`` where
values are numeric (int/float) and finite.

Generating JSON from Coq
------------------------
1. Make sure your Coq file defines the constants you need (e.g., ``C_tail`` and
   ``C_res``). From the repository root, run:

   ``coqtop -batch -quiet -l path/to/ulelat.v -eval "Print C_tail." -eval "Print C_res."``

   The output will include lines such as ``C_tail = 1.0``.

2. Capture the values into JSON using Python (example):

   ``coqtop -batch -quiet -l path/to/ulelat.v -eval "Print C_tail." -eval "Print C_res." > /tmp/coq_constants.txt``
   ``python - <<'PY'\nimport json,re,math\ntext=open('/tmp/coq_constants.txt').read()\npat=re.compile(r"(?P<name>[A-Za-z0-9_']+)\s*[:=].*?(?P<val>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")\nconsts={m.group('name'): float(m.group('val')) for m in pat.finditer(text)}\njson.dump(consts, open('uelat_constants.json','w'), indent=2)\nPY``

The helper script ``tools/generate_uelat_constants_from_coq.sh`` automates this
flow and adds error handling.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import warnings
from typing import Dict

logger = logging.getLogger(__name__)

_DEFAULT_CONSTANTS = {
    "C_tail": 1.0,
    "C_res": 1.0,
}

_constants: Dict[str, float] = dict(_DEFAULT_CONSTANTS)


class ConstantLoadError(RuntimeError):
    """Raised when constants cannot be parsed from an input artifact."""


def _validate_constants(data: Dict[str, float]) -> Dict[str, float]:
    validated: Dict[str, float] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise ConstantLoadError("Constant names must be strings; got %r" % (key,))
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ConstantLoadError(f"Value for constant '{key}' is not numeric: {value!r}") from exc
        if not math.isfinite(numeric):
            raise ConstantLoadError(f"Value for constant '{key}' is not finite: {numeric!r}")
        validated[key] = numeric
    return validated


def load_constants_from_json(path: str) -> Dict[str, float]:
    """Load constants from a JSON file.

    The JSON schema must be ``{"name": number, ...}``. Keys are strings and
    values must be finite numbers. Missing constants fall back to conservative
    defaults with a warning. Loaded constants override the module-level
    registry used by :func:`get_constant`.
    """

    expanded = os.path.expanduser(os.path.expandvars(path))
    if not os.path.exists(expanded):
        warnings.warn(
            f"Constants JSON '{path}' not found; falling back to defaults {_DEFAULT_CONSTANTS}",
            RuntimeWarning,
        )
        return dict(_constants)
    with open(expanded, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ConstantLoadError(f"Failed to parse JSON constants from {path}: {exc}") from exc
    validated = _validate_constants(data)
    missing = [k for k in _DEFAULT_CONSTANTS if k not in validated]
    if missing:
        warnings.warn(
            f"Constants JSON missing {missing}; using conservative defaults for them",
            RuntimeWarning,
        )
        for name in missing:
            validated[name] = _DEFAULT_CONSTANTS[name]
    _constants.clear()
    _constants.update(validated)
    return dict(_constants)


_COQ_PATTERN = re.compile(
    r"(?P<name>[A-Za-z0-9_']+)\s*(?:[:=]|:=).*?(?P<value>[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.MULTILINE,
)


def load_constants_from_coq_output(path: str) -> Dict[str, float]:
    """Parse constants from a Coq output text file.

    Parameters
    ----------
    path: str
        Path to a text file produced by ``coqtop -batch`` with ``Print``
        statements for the desired constants.

    Returns
    -------
    dict
        Mapping of constants.

    Raises
    ------
    ConstantLoadError
        If parsing fails. The error message includes guidance on generating a
        JSON file using ``coqtop`` and Python.
    """

    expanded = os.path.expanduser(os.path.expandvars(path))
    if not os.path.exists(expanded):
        raise ConstantLoadError(f"Coq output file '{path}' does not exist")

    text = open(expanded, "r", encoding="utf-8").read()
    matches = list(_COQ_PATTERN.finditer(text))
    if not matches:
        raise ConstantLoadError(
            "Could not parse any constants from Coq output. "
            "Ensure the file contains lines like 'C_tail = 1.0'. "
            "As a fallback, generate JSON via: "
            "coqtop -batch -quiet -l <file.v> -eval 'Print C_tail.' > coq.txt && "
            "python - <<'PY'\nimport json,re\ntext=open('coq.txt').read()\n"
            "pat=re.compile(r\"(?P<name>[A-Za-z0-9_']+)\\s*[:=].*?(?P<val>[+-]?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)\")\n"
            "json.dump({m.group('name'): float(m.group('val')) for m in pat.finditer(text)}, open('uelat_constants.json','w'), indent=2)\nPY"
        )

    parsed = {m.group("name"): float(m.group("value")) for m in matches}
    validated = _validate_constants(parsed)
    _constants.clear()
    _constants.update(validated)
    missing = [k for k in _DEFAULT_CONSTANTS if k not in validated]
    if missing:
        warnings.warn(
            f"Parsed Coq output but missing {missing}; falling back to defaults for them",
            RuntimeWarning,
        )
        for name in missing:
            _constants[name] = _DEFAULT_CONSTANTS[name]
    return dict(_constants)


def get_constant(name: str) -> float:
    """Return a constant by name, falling back to a conservative default.

    If the requested constant was not loaded, a warning is emitted and the
    default value is returned. This guarantees downstream code remains
    conservative even with incomplete inputs.
    """

    if name in _constants:
        value = _constants[name]
    else:
        warnings.warn(
            f"Constant '{name}' not loaded; returning conservative default 1.0",
            RuntimeWarning,
        )
        value = 1.0
        _constants[name] = value
    if not math.isfinite(value):  # pragma: no cover - defensive
        raise ConstantLoadError(f"Constant '{name}' is not finite: {value!r}")
    return float(value)


def list_constants() -> Dict[str, float]:
    """Return a shallow copy of the currently loaded constants."""

    return dict(_constants)


__all__ = [
    "load_constants_from_json",
    "load_constants_from_coq_output",
    "get_constant",
    "list_constants",
    "ConstantLoadError",
]
