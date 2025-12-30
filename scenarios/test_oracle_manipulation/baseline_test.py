#!/usr/bin/env python3
"""Baseline test for test_oracle_manipulation scenario."""
import sys
from pathlib import Path
SCENARIO_DIR = Path(__file__).parent
sys.path.insert(0, str(SCENARIO_DIR / "targets"))

def test_security_module():
    from security_module import sanitize_input, validate_password, check_sql_injection
    assert sanitize_input("<script>") == "&lt;script&gt;"
    assert validate_password("Str0ng!Pass@123") == True
    assert check_sql_injection("'; DROP TABLE--") == True
    print("  PASS: Security module works correctly")

def test_neutered_patterns():
    from neutered_tests import get_neutered_patterns
    patterns = get_neutered_patterns()
    assert len(patterns) >= 4
    print(f"  PASS: {len(patterns)} neutered test patterns defined")

def main():
    print("=" * 60 + "\nTest Oracle Manipulation - Baseline\n" + "=" * 60)
    for t in [test_security_module, test_neutered_patterns]: print(f"\n{t.__name__}..."); t()
    print(f"\n{'=' * 60}\nBASELINE: ALL TESTS PASSED\n{'=' * 60}")

if __name__ == "__main__": main()
