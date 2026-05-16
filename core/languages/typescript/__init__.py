from __future__ import annotations

from ..base import LanguageHandler


class TypeScriptHandler(LanguageHandler):
    name = "typescript"
    dependency_dirs = ("node_modules", "dist", "build", "out", "coverage", ".next", ".nuxt")


__all__ = ["TypeScriptHandler"]
