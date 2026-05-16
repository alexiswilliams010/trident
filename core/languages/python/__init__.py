from __future__ import annotations

from ..base import ImportEntry, LanguageHandler
from .imports import extract_python_imports
from .index import PythonIndexState, python_dotted_for


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

    def init_state(self) -> PythonIndexState:
        return PythonIndexState()

    def index_file(self, fvid: int, rel_path: str, state: PythonIndexState) -> None:
        dotted = python_dotted_for(rel_path)
        if dotted:
            state.pkg_index[dotted] = fvid


__all__ = ["PythonHandler"]
