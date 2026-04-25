"""Uses relative imports — exercise dot-counting in Phase 3."""

from .utils import helper


def relative_call(x: int) -> int:
    return helper(x)
