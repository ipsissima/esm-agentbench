#!/usr/bin/env python3
"""Smoke test for judge mode.

This test verifies that the judge mode entrypoint can run successfully
(or at least identifies missing dependencies gracefully).

Note: This is a smoke test only. It may be skipped in CI if models
are not available.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


class TestJudgeModeSmokeTest:
    """Smoke tests for judge mode execution."""

    @pytest.mark.slow
    def test_judge_mode_script_exists(self):
        """Judge mode script must exist."""
        judge_script = PROJECT_ROOT / "run_judge_mode.py"
        assert judge_script.exists(), "run_judge_mode.py not found"

    @pytest.mark.slow
    def test_judge_mode_help(self):
        """Judge mode should display help without errors."""
        judge_script = PROJECT_ROOT / "run_judge_mode.py"

        result = subprocess.run(
            [sys.executable, str(judge_script), "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Help failed: {result.stderr}"
        assert "judge mode" in result.stdout.lower()

    @pytest.mark.slow
    @pytest.mark.skipif(
        not (PROJECT_ROOT / "tools" / "real_agents_hf" / "models.yaml").exists(),
        reason="Models config not found, likely missing dependencies"
    )
    def test_real_agent_runner_exists(self):
        """Real agent runner must exist."""
        runner = PROJECT_ROOT / "tools" / "real_agents_hf" / "run_real_agents.py"
        assert runner.exists(), "run_real_agents.py not found"

    def test_scenario_structure_valid(self):
        """Scenarios must have required structure for judge mode."""
        scenarios_dir = PROJECT_ROOT / "scenarios"
        assert scenarios_dir.exists(), "scenarios/ directory not found"

        # Check at least one scenario
        code_backdoor = scenarios_dir / "code_backdoor_injection"
        assert code_backdoor.exists(), "code_backdoor_injection scenario not found"

        # Required files
        assert (code_backdoor / "plugin.py").exists(), "plugin.py not found"
        assert (code_backdoor / "baseline_test.py").exists(), "baseline_test.py not found"

        # Real agent prompts
        prompts_dir = code_backdoor / "real_agent_prompts"
        assert prompts_dir.exists(), "real_agent_prompts/ not found"
        assert (prompts_dir / "gold.md").exists(), "gold.md not found"
        assert (prompts_dir / "creative.md").exists(), "creative.md not found"
        assert (prompts_dir / "drift.md").exists(), "drift.md not found"

    def test_analysis_script_exists(self):
        """Analysis script must exist."""
        analysis_script = PROJECT_ROOT / "analysis" / "run_real_hf_experiment.py"
        assert analysis_script.exists(), "run_real_hf_experiment.py not found"

    def test_evaluation_config_exists(self):
        """Evaluation config must exist."""
        config = PROJECT_ROOT / "evaluation_config.yaml"
        assert config.exists(), "evaluation_config.yaml not found"

        # Should be valid YAML
        import yaml
        with open(config) as f:
            data = yaml.safe_load(f)

        assert isinstance(data, dict), "Config must be a dict"
        assert "certificate" in data, "Config must have certificate section"
        assert "pca_rank" in data["certificate"], "Config must specify pca_rank"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
