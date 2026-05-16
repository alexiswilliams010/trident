from __future__ import annotations

from ..base import ImportEntry, LanguageHandler
from .imports import extract_python_imports


class PythonHandler(LanguageHandler):
    name = "python"
    dependency_dirs = ("venv", ".venv", "site-packages", "env", ".env")

    def extract_imports(
        self,
        file_version_id: int,
        source_rel_path: str,
        ts_root,
        db_id_for: dict[int, int],
    ) -> list[ImportEntry]:
        return extract_python_imports(file_version_id, source_rel_path, ts_root, db_id_for)


__all__ = ["PythonHandler"]
