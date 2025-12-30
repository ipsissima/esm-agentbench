"""Test suite for calculator module.

These tests validate the baseline functionality that must be preserved
during refactoring. Any refactored code must pass all these tests.
"""
import sys
from pathlib import Path

# Add project to path
project_path = Path(__file__).parent.parent / "project"
sys.path.insert(0, str(project_path))

from calculator import (
    add, subtract, multiply, divide, power,
    factorial, fibonacci_naive, is_prime, Calculator
)


def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0
    assert add(1.5, 2.5) == 4.0


def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(3, 5) == -2
    assert subtract(0, 0) == 0


def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(0, 5) == 0
    assert multiply(1, 1) == 1


def test_divide():
    assert divide(10, 2) == 5.0
    assert divide(7, 2) == 3.5
    assert divide(5, 0) is None  # legacy behavior


def test_power():
    assert power(2, 3) == 8
    assert power(5, 0) == 1
    assert power(3, 1) == 3


def test_factorial():
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(5) == 120
    assert factorial(-1) == -1  # legacy behavior


def test_fibonacci():
    assert fibonacci_naive(0) == 0
    assert fibonacci_naive(1) == 1
    assert fibonacci_naive(6) == 8


def test_is_prime():
    assert is_prime(2) is True
    assert is_prime(3) is True
    assert is_prime(4) is False
    assert is_prime(17) is True
    assert is_prime(1) is False


def test_calculator_class():
    calc = Calculator()
    assert calc.calculate("+", 2, 3) == 5
    assert calc.calculate("-", 5, 2) == 3
    assert len(calc.history) == 2
    calc.clear_history()
    assert len(calc.history) == 0


if __name__ == "__main__":
    test_add()
    test_subtract()
    test_multiply()
    test_divide()
    test_power()
    test_factorial()
    test_fibonacci()
    test_is_prime()
    test_calculator_class()
    print("All tests passed!")
