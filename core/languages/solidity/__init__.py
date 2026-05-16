from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import ImportEntry, LanguageHandler, ResolvedImport
from .imports import extract_solidity_imports
from .resolve import resolve_solidity

if TYPE_CHECKING:
    from ...config_loader import LanguageConfig
    from ...heuristic_resolver import BranchIndex


class SolidityHandler(LanguageHandler):
    name = "solidity"
    dependency_dirs = ("lib", "node_modules", "out", "cache", "artifacts")

    def extract_imports(
        self,
        file_version_id: int,
        source_rel_path: str,
        ts_root,
        db_id_for: dict[int, int],
    ) -> list[ImportEntry]:
        return extract_solidity_imports(file_version_id, source_rel_path, ts_root, db_id_for)

    def resolve(
        self, entry: ImportEntry, idx: "BranchIndex", cfg: "LanguageConfig",
    ) -> ResolvedImport:
        return resolve_solidity(entry, idx, cfg)


__all__ = ["SolidityHandler"]
