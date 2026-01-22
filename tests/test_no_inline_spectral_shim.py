import pathlib
import re

import pytest

# Mark all tests in this module as unit tests (Tier 1)
pytestmark = pytest.mark.unit

SHIM_PATTERNS = [
    r"\bnp\.linalg\.pinv\b",
    r"\bkoopman_residual\b",
    r"\bspectral_bound\s*=",
    r"compute_spectral_features_from_trace\s*\(",
]

SCENARIO_DIR = pathlib.Path(__file__).resolve().parent.parent / "scenarios"


def test_no_inline_shim() -> None:
    files = list(SCENARIO_DIR.glob("*/plugin.py"))
    offenders = {}
    for file_path in files:
        text = file_path.read_text()
        for pat in SHIM_PATTERNS:
            if re.search(pat, text):
                offenders.setdefault(str(file_path), []).append(pat)
    if offenders:
        msg_lines = ["Inline spectral shim found in scenario plugins:"]
        for name, patterns in offenders.items():
            msg_lines.append(f"{name}: {patterns}")
        pytest.fail("\n".join(msg_lines))
