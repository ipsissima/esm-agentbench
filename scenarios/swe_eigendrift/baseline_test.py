#!/usr/bin/env python3
"""Baseline test for swe_eigendrift scenario.

This test validates that the target calculator module functions correctly
and that the test suite passes. The baseline must succeed before any
attack can be meaningfully evaluated.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCENARIO_DIR = Path(__file__).parent
TARGETS_DIR = SCENARIO_DIR / "targets"
PROJECT_DIR = TARGETS_DIR / "project"
TESTS_DIR = TARGETS_DIR / "tests"

# Add project to path
sys.path.insert(0, str(PROJECT_DIR))


def run_calculator_tests() -> bool:
    """Run all calculator tests and return success status."""
    from targets.tests.test_calculator import (
        test_add, test_subtract, test_multiply, test_divide,
        test_power, test_factorial, test_fibonacci, test_is_prime,
        test_calculator_class
    )

    tests = [
        ("test_add", test_add),
        ("test_subtract", test_subtract),
        ("test_multiply", test_multiply),
        ("test_divide", test_divide),
        ("test_power", test_power),
        ("test_factorial", test_factorial),
        ("test_fibonacci", test_fibonacci),
        ("test_is_prime", test_is_prime),
        ("test_calculator_class", test_calculator_class),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
            print(f"  PASS: {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {name} - {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR: {name} - {e}")

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_imports_work():
    """Test that all target modules can be imported."""
    from targets.project.calculator import (
        add, subtract, multiply, divide, power,
        factorial, fibonacci_naive, is_prime, Calculator
    )
    assert callable(add)
    assert callable(Calculator)
    print("  PASS: All imports successful")


def test_basic_sanity():
    """Run basic sanity checks on calculator functions."""
    from targets.project.calculator import add, multiply

    # Basic arithmetic
    assert add(1, 1) == 2, "add(1,1) should equal 2"
    assert multiply(2, 3) == 6, "multiply(2,3) should equal 6"
    print("  PASS: Basic sanity checks")


def main():
    """Run all baseline tests."""
    print("=" * 60)
    print("SWE EigenDrift Scenario - Baseline Tests")
    print("=" * 60)

    print("\n1. Testing imports...")
    test_imports_work()

    print("\n2. Running sanity checks...")
    test_basic_sanity()

    print("\n3. Running full calculator test suite...")
    success = run_calculator_tests()

    print("\n" + "=" * 60)
    if success:
        print("BASELINE: ALL TESTS PASSED")
    else:
        print("BASELINE: SOME TESTS FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
