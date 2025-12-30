#!/usr/bin/env python3
"""Tests for AgentBeats Phase-1 submission validation.

This test module validates that the repository meets all Phase-1 requirements:
1. Each scenario has attack_succeeded.json with success:true
2. Baseline tests pass for each scenario
3. Plugin produces valid output with evidence
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SCENARIOS_DIR = REPO_ROOT / "scenarios"

# Expected scenarios for Phase-1 (6 coding-agent scenarios)
EXPECTED_SCENARIOS = [
    "code_backdoor_injection",
    "supply_chain_poisoning",
    "test_oracle_manipulation",
    "code_review_bypass",
    "debug_credential_leak",
    "refactor_vuln_injection",
]


def test_scenarios_directory_exists():
    """Verify scenarios/ directory exists."""
    assert SCENARIOS_DIR.exists(), "scenarios/ directory must exist"
    assert SCENARIOS_DIR.is_dir(), "scenarios/ must be a directory"


def test_all_expected_scenarios_present():
    """Verify all 6 expected scenarios are present."""
    scenario_dirs = [d.name for d in SCENARIOS_DIR.iterdir() if d.is_dir()]
    for expected in EXPECTED_SCENARIOS:
        assert expected in scenario_dirs, f"Scenario '{expected}' must be present"


@pytest.mark.parametrize("scenario", EXPECTED_SCENARIOS)
def test_scenario_has_required_files(scenario: str):
    """Verify each scenario has all required files."""
    scenario_dir = SCENARIOS_DIR / scenario
    required_files = [
        "manifest.json",
        "baseline_test.py",
        "plugin.py",
        "README.md",
    ]
    for fname in required_files:
        fpath = scenario_dir / fname
        assert fpath.exists(), f"{scenario} missing required file: {fname}"


@pytest.mark.parametrize("scenario", EXPECTED_SCENARIOS)
def test_scenario_has_targets(scenario: str):
    """Verify each scenario has a targets/ directory with content."""
    targets_dir = SCENARIOS_DIR / scenario / "targets"
    assert targets_dir.exists(), f"{scenario} missing targets/ directory"
    assert targets_dir.is_dir(), f"{scenario}/targets must be a directory"
    files = list(targets_dir.iterdir())
    assert len(files) > 0, f"{scenario}/targets must contain files"


@pytest.mark.parametrize("scenario", EXPECTED_SCENARIOS)
def test_manifest_valid(scenario: str):
    """Verify each scenario's manifest.json is valid."""
    manifest_path = SCENARIOS_DIR / scenario / "manifest.json"
    with manifest_path.open() as f:
        manifest = json.load(f)

    required_keys = ["id", "title", "difficulty", "type", "rounds_allowed", "success_criterion"]
    for key in required_keys:
        assert key in manifest, f"{scenario}/manifest.json missing key: {key}"

    assert manifest["rounds_allowed"] <= 10, f"{scenario} rounds_allowed must be <= 10"
    assert manifest["id"] == scenario, f"{scenario} manifest id must match directory name"


@pytest.mark.parametrize("scenario", EXPECTED_SCENARIOS)
def test_baseline_runs(scenario: str):
    """Verify baseline_test.py runs successfully for each scenario."""
    scenario_dir = SCENARIOS_DIR / scenario
    result = subprocess.run(
        [sys.executable, str(scenario_dir / "baseline_test.py")],
        capture_output=True,
        text=True,
        cwd=str(scenario_dir),
        timeout=60,
    )
    assert result.returncode == 0, f"{scenario} baseline failed:\n{result.stdout}\n{result.stderr}"


@pytest.mark.parametrize("scenario", EXPECTED_SCENARIOS)
def test_plugin_runs_and_produces_attack_succeeded(scenario: str):
    """Verify plugin.py runs and produces attack_succeeded.json."""
    scenario_dir = SCENARIOS_DIR / scenario

    # Run plugin
    result = subprocess.run(
        [sys.executable, str(scenario_dir / "plugin.py")],
        capture_output=True,
        text=True,
        cwd=str(scenario_dir),
        timeout=120,
    )
    assert result.returncode == 0, f"{scenario} plugin failed:\n{result.stdout}\n{result.stderr}"

    # Check attack_succeeded.json exists
    attack_file = scenario_dir / "attack_succeeded.json"
    assert attack_file.exists(), f"{scenario} missing attack_succeeded.json after plugin run"


@pytest.mark.parametrize("scenario", EXPECTED_SCENARIOS)
def test_attack_succeeded_is_true(scenario: str):
    """Verify attack_succeeded.json has success:true and valid evidence."""
    scenario_dir = SCENARIOS_DIR / scenario
    attack_file = scenario_dir / "attack_succeeded.json"

    # Run plugin first to ensure file exists
    subprocess.run(
        [sys.executable, str(scenario_dir / "plugin.py")],
        capture_output=True,
        cwd=str(scenario_dir),
        timeout=120,
    )

    assert attack_file.exists(), f"{scenario} missing attack_succeeded.json"

    with attack_file.open() as f:
        result = json.load(f)

    # Gate check: success must be true
    assert result.get("success") is True, f"{scenario} attack_succeeded.json success != true"

    # Evidence must be present
    evidence = result.get("evidence")
    assert evidence is not None, f"{scenario} missing evidence in attack_succeeded.json"
    assert "trace_path" in evidence, f"{scenario} evidence missing trace_path"
    assert "snippet" in evidence, f"{scenario} evidence missing snippet"

    # Snippet should be non-empty
    assert len(evidence["snippet"]) > 0, f"{scenario} evidence snippet is empty"


def test_agent_card_is_actionable():
    """Verify agent_card.toml has actionable entrypoint."""
    import tomli

    agent_card_path = REPO_ROOT / "agent_card.toml"
    assert agent_card_path.exists(), "agent_card.toml must exist"

    with agent_card_path.open("rb") as f:
        agent_card = tomli.load(f)

    entrypoint = agent_card.get("entrypoint", "")
    assert "LOCAL" in entrypoint or "http" in entrypoint, \
        "agent_card.toml entrypoint must be actionable (contain LOCAL or http)"

    # Check for agentbeats section
    assert "agentbeats" in agent_card, "agent_card.toml should have [agentbeats] section"
    assert agent_card["agentbeats"].get("phase") == 1, "agentbeats.phase should be 1"


def test_readme_for_each_scenario():
    """Verify each scenario README has required sections."""
    required_sections = [
        "Overview",
        "Success Criterion",
        "How to Run",
        "Evidence",
        "Novelty",
    ]

    for scenario in EXPECTED_SCENARIOS:
        readme_path = SCENARIOS_DIR / scenario / "README.md"
        assert readme_path.exists(), f"{scenario} missing README.md"

        content = readme_path.read_text()
        for section in required_sections:
            # Check for section header (## or #)
            assert section in content, f"{scenario}/README.md missing section: {section}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
