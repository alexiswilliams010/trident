from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..base import ImportEntry, LanguageHandler, ResolvedImport
from .crates import RustIndexState, finalize_rust_state
from .imports import extract_rust_imports
from .resolve import resolve_rust

if TYPE_CHECKING:
    from ...config_loader import LanguageConfig
    from ...heuristic_resolver import BranchIndex


class RustHandler(LanguageHandler):
    name = "rust"

    def extract_imports(
        self,
        file_version_id: int,
        source_rel_path: str,
        ts_root,
        db_id_for: dict[int, int],
    ) -> list[ImportEntry]:
        return extract_rust_imports(file_version_id, source_rel_path, ts_root, db_id_for)

    def init_state(self) -> RustIndexState:
        return RustIndexState()

    def index_file(self, fvid: int, rel_path: str, state: RustIndexState) -> None:
        # Crate ownership requires the full crate set, which isn't known until
        # finalize_index. Buffer the file here; finalize resolves owners.
        state.pending_files.append((fvid, rel_path))

    def finalize_index(self, repo_root: Path | None, state: RustIndexState) -> None:
        finalize_rust_state(repo_root, state)

    def resolve(
        self, entry: ImportEntry, idx: "BranchIndex", cfg: "LanguageConfig",
    ) -> ResolvedImport:
        return resolve_rust(entry, idx)


__all__ = ["RustHandler"]
