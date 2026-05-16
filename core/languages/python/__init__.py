from __future__ import annotations

from ..base import LanguageHandler


class PythonHandler(LanguageHandler):
    name = "python"
    dependency_dirs = ("venv", ".venv", "site-packages", "env", ".env")


__all__ = ["PythonHandler"]
