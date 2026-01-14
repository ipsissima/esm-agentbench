from pathlib import Path


def test_no_wildcard_imports_in_scenarios():
    scenario_paths = list(Path("scenarios").rglob("*.py"))
    for path in scenario_paths:
        content = path.read_text(encoding="utf-8")
        assert "from solution import *" not in content, f"Wildcard import found in {path}"
