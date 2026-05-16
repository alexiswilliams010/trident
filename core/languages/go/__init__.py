from __future__ import annotations

from pathlib import Path

from ..base import ImportEntry, LanguageHandler
from .imports import extract_go_imports
from .index import GoIndexState, finalize_go_state, index_go_file


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

    def init_state(self) -> GoIndexState:
        return GoIndexState()

    def index_file(self, fvid: int, rel_path: str, state: GoIndexState) -> None:
        index_go_file(fvid, rel_path, state)

    def finalize_index(self, repo_root: Path | None, state: GoIndexState) -> None:
        finalize_go_state(repo_root, state)


__all__ = ["GoHandler"]
