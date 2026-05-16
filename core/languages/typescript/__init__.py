from __future__ import annotations

from pathlib import Path

from .._shared.node_imports import extract_node_imports
from .._shared.node_index import NodeIndexState, finalize_node_state
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

    def init_state(self) -> NodeIndexState:
        return NodeIndexState()

    def finalize_index(self, repo_root: Path | None, state: NodeIndexState) -> None:
        finalize_node_state(repo_root, state)


__all__ = ["TypeScriptHandler"]
