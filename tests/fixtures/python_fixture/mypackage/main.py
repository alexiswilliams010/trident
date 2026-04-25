"""Entrypoint demonstrating intra-package + external imports."""

from mypackage.utils import double, helper

import requests  # external — should classify as external in Phase 3


GREETING = "hello"


class Calculator:
    def __init__(self, base: int) -> None:
        self.base = base

    def add(self, x: int) -> int:
        return self.base + helper(x)

    def double_it(self, x: int) -> int:
        return double(x)


def run() -> int:
    calc = Calculator(10)
    return calc.add(5)
