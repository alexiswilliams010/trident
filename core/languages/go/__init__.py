from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..base import ImportEntry, LanguageHandler, ResolvedImport
from .imports import extract_go_imports
from .index import GoIndexState, finalize_go_state, index_go_file
from .resolve import resolve_go

if TYPE_CHECKING:
    from ...config_loader import LanguageConfig
    from ...heuristic_resolver import BranchIndex


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

    def resolve(
        self, entry: ImportEntry, idx: "BranchIndex", cfg: "LanguageConfig",
    ) -> ResolvedImport:
        return resolve_go(entry, idx)

    def seed_implicit_imports(self, idx: "BranchIndex") -> dict[int, set[int]]:
        """Go: every file in a package implicitly imports its peers."""
        state = idx.lang_state.get("go")
        if state is None:
            return {}
        out: dict[int, set[int]] = {}
        for peers in state.pkg_files.values():
            peer_set = set(peers)
            for fvid in peers:
                out.setdefault(fvid, set()).update(peer_set - {fvid})
        return out

    def expand_import_target(
        self, target_fvid: int, idx: "BranchIndex",
    ) -> set[int]:
        """A Go `import "x/y/z"` references a package, not a single file.
        Expand to every sibling .go file in the imported package directory
        so the cross-file linker can match symbols defined in any peer."""
        state = idx.lang_state.get("go")
        if state is None:
            return {target_fvid}
        target_path = idx.files_by_id.get(target_fvid, "")
        pkg_dir = "/".join(target_path.split("/")[:-1])
        out = {target_fvid}
        out.update(state.pkg_files.get(pkg_dir, []))
        return out


__all__ = ["GoHandler"]
