from __future__ import annotations

from ..base import LanguageHandler


class JavaScriptHandler(LanguageHandler):
    name = "javascript"
    dependency_dirs = ("node_modules", "dist", "build", "out", "coverage", ".next", ".nuxt")


__all__ = ["JavaScriptHandler"]
