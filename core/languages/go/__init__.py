from __future__ import annotations

from ..base import LanguageHandler


class GoHandler(LanguageHandler):
    name = "go"
    dependency_dirs = ("vendor",)


__all__ = ["GoHandler"]
