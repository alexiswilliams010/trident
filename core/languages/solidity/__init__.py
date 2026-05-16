from __future__ import annotations

from ..base import LanguageHandler


class SolidityHandler(LanguageHandler):
    name = "solidity"
    dependency_dirs = ("lib", "node_modules", "out", "cache", "artifacts")


__all__ = ["SolidityHandler"]
