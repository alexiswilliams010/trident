"""Inheritance fixture: same-file + cross-file class hierarchies."""

from mypackage.utils import helper


class BasePolicy:
    def is_active(self) -> bool:
        return True

    def check(self, x: int) -> int:
        return helper(x)


class StrictPolicy(BasePolicy):
    """Same-file inheritance + method override."""

    def is_active(self) -> bool:
        return False

    def check(self, x: int) -> int:
        return super().check(x) * 2
