from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..base import (
    ImportEntry,
    InheritanceEdge,
    LanguageHandler,
    ResolvedImport,
    ResolvedReference,
    SemanticContext,
)
from .crates import RustIndexState, finalize_rust_state
from .imports import extract_rust_imports
from .resolve import resolve_rust
from .semantic import (
    collect_rust_impl_method_targets,
    collect_rust_test_skip_ids,
    is_rust_test_path,
    resolve_rust_reference,
    rust_qualified_name_prefix,
    synthesize_rust_inheritance,
)

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

    # ── semantic hooks ───────────────────────────────────────────────
    def should_skip_file(self, rel_path: str) -> bool:
        return is_rust_test_path(rel_path)

    def precompute_file_state(self, ts_walk: list, ctx: SemanticContext) -> None:
        ctx.scratch["test_skip_ts_ids"] = collect_rust_test_skip_ids(ts_walk)
        ctx.scratch["impl_method_target"] = collect_rust_impl_method_targets(ts_walk)

    def resolve_reference(
        self, ts_node, rule, ctx: SemanticContext,
    ) -> ResolvedReference | None:
        return resolve_rust_reference(ts_node, rule, ctx)

    def qualified_name_prefix(self, ts_node, ctx: SemanticContext) -> str | None:
        return rust_qualified_name_prefix(ts_node, ctx)

    def synthesize_inheritance(
        self, ts_walk: list, ctx: SemanticContext,
    ) -> list[InheritanceEdge]:
        return synthesize_rust_inheritance(ts_walk, ctx)


__all__ = ["RustHandler"]
