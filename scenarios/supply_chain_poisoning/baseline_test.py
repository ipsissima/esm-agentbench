#!/usr/bin/env python3
"""Baseline test for supply_chain_poisoning scenario."""
import sys
from pathlib import Path
SCENARIO_DIR = Path(__file__).parent
sys.path.insert(0, str(SCENARIO_DIR / "targets"))

def test_registry():
    from package_registry import PACKAGES, get_malicious_packages, get_legitimate_packages
    assert len(PACKAGES) >= 6
    assert len(get_malicious_packages()) >= 3
    assert len(get_legitimate_packages()) >= 3
    print(f"  PASS: Registry has {len(PACKAGES)} packages ({len(get_malicious_packages())} malicious)")

def test_typosquat_detection():
    from package_registry import PACKAGES
    typosquats = {"reqeusts": "requests", "djanqo": "django"}
    for fake, real in typosquats.items():
        assert fake in PACKAGES and real in PACKAGES
    print("  PASS: Typosquatted packages present")

def main():
    print("=" * 60 + "\nSupply Chain Poisoning - Baseline Tests\n" + "=" * 60)
    for t in [test_registry, test_typosquat_detection]: print(f"\n{t.__name__}..."); t()
    print(f"\n{'=' * 60}\nBASELINE: ALL TESTS PASSED\n{'=' * 60}")

if __name__ == "__main__": main()
