from __future__ import annotations

from ..base import ImportEntry, LanguageHandler
from .imports import extract_go_imports


class GoHandler(LanguageHandler):
    name = "go"
    dependency_dirs = ("vendor",)

    def extract_imports(
        self,
        file_version_id: int,
        source_rel_path: str,
        ts_root,
        db_id_for: dict[int, int],
    ) -> list[ImportEntry]:
        return extract_go_imports(file_version_id, source_rel_path, ts_root, db_id_for)


__all__ = ["GoHandler"]
