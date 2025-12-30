#!/usr/bin/env python3
"""Baseline test for code_backdoor_injection scenario."""
import sys
from pathlib import Path

SCENARIO_DIR = Path(__file__).parent
TARGETS_DIR = SCENARIO_DIR / "targets"

def test_targets_exist():
    assert (TARGETS_DIR / "auth_module.py").exists()
    assert (TARGETS_DIR / "adversarial_context.py").exists()
    print("  PASS: Target files exist")

def test_backdoor_patterns():
    sys.path.insert(0, str(TARGETS_DIR))
    from adversarial_context import get_backdoor_patterns
    patterns = get_backdoor_patterns()
    assert len(patterns) >= 4
    print(f"  PASS: {len(patterns)} backdoor detection patterns loaded")

def test_auth_module_structure():
    content = (TARGETS_DIR / "auth_module.py").read_text()
    assert "class AuthService" in content
    assert "def authenticate" in content
    print("  PASS: Auth module has correct structure")

def main():
    print("=" * 60)
    print("Code Backdoor Injection - Baseline Tests")
    print("=" * 60)
    tests = [test_targets_exist, test_backdoor_patterns, test_auth_module_structure]
    passed = sum(1 for t in tests if not (print(f"\n{t.__name__}...") or t() or False) or True)
    print(f"\n{'=' * 60}\nBASELINE: ALL TESTS PASSED\n{'=' * 60}")

if __name__ == "__main__":
    main()
