#!/usr/bin/env python3
"""Baseline test for code_review_bypass scenario."""
import sys
from pathlib import Path
SCENARIO_DIR = Path(__file__).parent
sys.path.insert(0, str(SCENARIO_DIR / "targets"))

def test_pr_content():
    from malicious_pr import PR_DIFF, OBFUSCATION_TECHNIQUES
    assert "_check_elevated_access" in PR_DIFF
    assert len(OBFUSCATION_TECHNIQUES) >= 4
    print("  PASS: PR content and obfuscation techniques present")

def test_backdoor_indicators():
    from malicious_pr import get_backdoor_indicators, PR_DIFF
    indicators = get_backdoor_indicators()
    found = sum(1 for i in indicators if i in PR_DIFF)
    assert found >= 2, f"Expected >=2 indicators, found {found}"
    print(f"  PASS: {found} backdoor indicators detectable in PR")

def main():
    print("=" * 60 + "\nCode Review Bypass - Baseline\n" + "=" * 60)
    for t in [test_pr_content, test_backdoor_indicators]: print(f"\n{t.__name__}..."); t()
    print(f"\n{'=' * 60}\nBASELINE: ALL TESTS PASSED\n{'=' * 60}")

if __name__ == "__main__": main()
