from __future__ import annotations

from .._shared.node_imports import extract_node_imports
from ..base import ImportEntry, LanguageHandler


class TypeScriptHandler(LanguageHandler):
    name = "typescript"
    dependency_dirs = ("node_modules", "dist", "build", "out", "coverage", ".next", ".nuxt")

    def extract_imports(
        self,
        file_version_id: int,
        source_rel_path: str,
        ts_root,
        db_id_for: dict[int, int],
    ) -> list[ImportEntry]:
        return extract_node_imports(
            "typescript", file_version_id, source_rel_path, ts_root, db_id_for,
        )


__all__ = ["TypeScriptHandler"]
