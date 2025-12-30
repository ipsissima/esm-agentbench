#!/usr/bin/env python3
"""Test that no synthetic trace generation exists in evidence paths.

This test ensures that synthetic trace generation code is isolated to
tests only and cannot leak into the evaluation or evidence pipeline.
"""
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent

# Evidence paths that must NOT contain synthetic generation
EVIDENCE_PATHS = [
    "analysis",
    "tools/real_agents_hf",
    "scenarios/*/plugin.py",
    "reports/spectral_validation_real_hf",
]

# Banned patterns indicating synthetic trace generation
BANNED_PATTERNS = [
    r"np\.random\.default_rng",
    r"np\.random\.RandomState",
    r"random\.seed\(",
    r"generate_.*_trace\(",
    r"def generate_.*trace",
    r"synthetic.*trace",
    r"poison_vector\s*=",
    r"backdoor_vector\s*=",
    r"fake.*trace",
    r"simulated.*trace",
]

# Allowlist: files that can mention these for documentation purposes only
ALLOWLIST_FILES = [
    "README.md",
    "REAL_AGENT_HF_EVAL.md",
    "__pycache__",
]


def should_check_file(file_path: Path) -> bool:
    """Determine if file should be checked.

    Parameters
    ----------
    file_path : Path
        File path to check

    Returns
    -------
    bool
        True if file should be scanned
    """
    # Skip allowlisted files
    for allowed in ALLOWLIST_FILES:
        if allowed in str(file_path):
            return False

    # Only check Python files
    return file_path.suffix == ".py"


def scan_file_for_patterns(file_path: Path, patterns: list) -> list:
    """Scan file for banned patterns.

    Parameters
    ----------
    file_path : Path
        File to scan
    patterns : list
        Regex patterns to search for

    Returns
    -------
    list
        List of (line_num, line, pattern) tuples for matches
    """
    matches = []

    try:
        content = file_path.read_text()
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith("#"):
                continue

            for pattern in patterns:
                if re.search(pattern, line):
                    matches.append((line_num, line.strip(), pattern))

    except (UnicodeDecodeError, IOError):
        pass  # Skip binary files

    return matches


class TestNoSyntheticInEvidencePath:
    """Ensure no synthetic trace generation in evidence paths."""

    def test_analysis_scripts_no_synthetic(self):
        """analysis/ scripts must not generate synthetic traces."""
        analysis_dir = PROJECT_ROOT / "analysis"
        if not analysis_dir.exists():
            pytest.skip("analysis/ directory not found")

        violations = []

        for py_file in analysis_dir.glob("*.py"):
            if not should_check_file(py_file):
                continue

            matches = scan_file_for_patterns(py_file, BANNED_PATTERNS)
            if matches:
                violations.append((py_file, matches))

        if violations:
            msg = "\n".join([
                f"\n{file}:\n" + "\n".join([
                    f"  Line {line_num}: {line} (matched: {pattern})"
                    for line_num, line, pattern in matches
                ])
                for file, matches in violations
            ])
            pytest.fail(f"Found synthetic patterns in analysis/:\n{msg}")

    def test_real_agents_hf_no_synthetic(self):
        """tools/real_agents_hf/ must not generate synthetic traces."""
        tools_dir = PROJECT_ROOT / "tools" / "real_agents_hf"
        if not tools_dir.exists():
            pytest.skip("tools/real_agents_hf/ directory not found")

        violations = []

        for py_file in tools_dir.glob("*.py"):
            if not should_check_file(py_file):
                continue

            matches = scan_file_for_patterns(py_file, BANNED_PATTERNS)
            if matches:
                violations.append((py_file, matches))

        if violations:
            msg = "\n".join([
                f"\n{file}:\n" + "\n".join([
                    f"  Line {line_num}: {line} (matched: {pattern})"
                    for line_num, line, pattern in matches
                ])
                for file, matches in violations
            ])
            pytest.fail(f"Found synthetic patterns in tools/real_agents_hf/:\n{msg}")

    def test_scenario_plugins_no_synthetic(self):
        """Scenario plugin.py files must not generate synthetic traces."""
        scenarios_dir = PROJECT_ROOT / "scenarios"
        if not scenarios_dir.exists():
            pytest.skip("scenarios/ directory not found")

        violations = []

        for plugin_file in scenarios_dir.glob("*/plugin.py"):
            if not should_check_file(plugin_file):
                continue

            matches = scan_file_for_patterns(plugin_file, BANNED_PATTERNS)
            if matches:
                violations.append((plugin_file, matches))

        if violations:
            msg = "\n".join([
                f"\n{file}:\n" + "\n".join([
                    f"  Line {line_num}: {line} (matched: {pattern})"
                    for line_num, line, pattern in matches
                ])
                for file, matches in violations
            ])
            pytest.fail(f"Found synthetic patterns in scenario plugins:\n{msg}")

    def test_reports_directory_no_synthetic_code(self):
        """reports/ must not contain Python files with synthetic generation."""
        reports_dir = PROJECT_ROOT / "reports"
        if not reports_dir.exists():
            # Reports may not exist yet, which is fine
            return

        violations = []

        for py_file in reports_dir.glob("**/*.py"):
            if not should_check_file(py_file):
                continue

            matches = scan_file_for_patterns(py_file, BANNED_PATTERNS)
            if matches:
                violations.append((py_file, matches))

        if violations:
            msg = "\n".join([
                f"\n{file}:\n" + "\n".join([
                    f"  Line {line_num}: {line} (matched: {pattern})"
                    for line_num, line, pattern in matches
                ])
                for file, matches in violations
            ])
            pytest.fail(f"Found synthetic patterns in reports/:\n{msg}")

    def test_submissions_no_synthetic_code(self):
        """submissions/ must not contain synthetic generation code."""
        submissions_dir = PROJECT_ROOT / "submissions"
        if not submissions_dir.exists():
            # Submissions may not exist yet
            return

        violations = []

        for py_file in submissions_dir.glob("**/*.py"):
            if not should_check_file(py_file):
                continue

            matches = scan_file_for_patterns(py_file, BANNED_PATTERNS)
            if matches:
                violations.append((py_file, matches))

        if violations:
            msg = "\n".join([
                f"\n{file}:\n" + "\n".join([
                    f"  Line {line_num}: {line} (matched: {pattern})"
                    for line_num, line, pattern in matches
                ])
                for file, matches in violations
            ])
            pytest.fail(f"Found synthetic patterns in submissions/:\n{msg}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
