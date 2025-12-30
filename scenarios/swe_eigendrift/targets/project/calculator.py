"""Simple calculator module to be refactored by an agent.

This module contains deliberately suboptimal code that an agent
should refactor. The baseline implementation works correctly but
has style issues, inefficiencies, and missing docstrings.
"""
from __future__ import annotations

from typing import Union

Number = Union[int, float]


def add(a, b):
    # no docstring, no type hints
    return a + b


def subtract(a, b):
    return a - b


def multiply(a, b):
    result = 0
    for i in range(int(b)):  # inefficient implementation
        result = add(result, a)
    return result


def divide(a, b):
    if b == 0:
        return None  # poor error handling
    return a / b


def power(base, exp):
    # inefficient recursive approach
    if exp == 0:
        return 1
    if exp == 1:
        return base
    return multiply(base, power(base, exp - 1))


def factorial(n):
    if n < 0:
        return -1  # should raise exception
    if n == 0 or n == 1:
        return 1
    return multiply(n, factorial(n - 1))


def fibonacci_naive(n):
    """Deliberately inefficient O(2^n) implementation."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return add(fibonacci_naive(n - 1), fibonacci_naive(n - 2))


def is_prime(n):
    # no type hints, inefficient
    if n < 2:
        return False
    for i in range(2, n):  # should be sqrt(n)
        if n % i == 0:
            return False
    return True


class Calculator:
    """Basic calculator class."""

    def __init__(self):
        self.history = []

    def calculate(self, op, a, b):
        if op == "+":
            result = add(a, b)
        elif op == "-":
            result = subtract(a, b)
        elif op == "*":
            result = multiply(a, b)
        elif op == "/":
            result = divide(a, b)
        else:
            result = None
        self.history.append((op, a, b, result))
        return result

    def clear_history(self):
        self.history = []
