"""Per-language handler protocol.

Each supported language registers exactly one `LanguageHandler` subclass.
Handlers own all language-specific edge cases — import extraction, import
resolution, branch-index contribution, and optional semantic-resolver hooks.
The general code paths in `core.heuristic_resolver` and `core.semantic_resolver`
dispatch through the handler registry (see `core.languages.HANDLERS`); no
`if language == "..."` branching belongs in those files.

Required hooks raise NotImplementedError on the base so forgetting to override
fails loudly. Optional hooks default to no-ops so simple languages opt in only
to what they need.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config_loader import LanguageConfig
    from ..grammar_meta import LanguageSpec
    from ..heuristic_resolver import BranchIndex


@dataclass
class ImportEntry:
    """One import statement worth of info, before resolution."""

    file_version_id: int        # importer file_version
    node_id: int                # DB id of the import node
    language: str
    source_rel_path: str        # importer's repo-relative path (within the branch)
    import_path: str            # raw text path: "mypackage.utils", "./Token.sol", "..", "@oz/..."
    imported_names: list[str]   # specific symbols imported (e.g. ["helper", "double"])
    is_relative: bool           # Python: starts with "." ; Solidity: starts with "./" or "../"
    dot_count: int = 0          # Python: leading dots in `from . import …`


@dataclass
class ResolvedImport:
    entry: ImportEntry
    dep_class: str              # 'intra_repo' | 'external' | 'unresolved'
    resolved_file_version_id: int | None = None
    package_name: str | None = None
    external_dep_id: int | None = None


@dataclass
class SemanticContext:
    """Per-file context passed to semantic-resolver hooks.

    `scratch` is a free-form slot handlers can write into during
    `precompute_file_state` and read back from later hooks within the same
    file (e.g. Rust stashes test-skip node IDs at `scratch['test_skip_ts_ids']`).
    `module_def_id` and `defs_by_scope_and_name` are populated by the resolver
    before `synthesize_inheritance` fires so handlers can resolve intra-file
    base-name lookups.
    """

    config: "LanguageConfig"
    file_path: str
    scratch: dict[str, Any] = field(default_factory=dict)
    module_def_id: int | None = None
    defs_by_scope_and_name: dict[tuple[int, str], int] = field(default_factory=dict)


@dataclass
class ResolvedReference:
    """Return value of `LanguageHandler.resolve_reference`.

    Carries an explicit confidence so a handler that resolves via AST
    context can upgrade the resulting row above the rule's declared
    confidence when its answer is type-aware rather than name-only.
    """
    target_def_id: int
    confidence: str = "certain"


@dataclass
class InheritanceEdge:
    """One inheritance edge a handler wants emitted into `inherits_edges`.
    `base_def_id` is filled when intra-file resolution succeeds; otherwise
    cross-file linking fills it in Phase 3. The caller assigns `ord` from a
    per-target counter so emission order is stable."""

    child_def_id: int
    base_name: str
    base_def_id: int | None = None
    confidence: str = "inferred"


class LanguageHandler(ABC):
    """Per-language edge cases live behind this interface.

    Subclasses set `name` (matching a key in `core.grammar_meta.LANGUAGES`) as a
    class attribute. The base `__init__` looks the spec up and stores it on the
    instance so handlers can read tree-sitter metadata without re-importing.
    """

    name: str = ""
    dependency_dirs: tuple[str, ...] = ()

    def __init__(self) -> None:
        if not self.name:
            raise TypeError(f"{type(self).__name__} must set `name`")
        from ..grammar_meta import LANGUAGES
        try:
            self.spec: "LanguageSpec" = LANGUAGES[self.name]
        except KeyError as e:
            raise KeyError(
                f"{type(self).__name__} declares name={self.name!r} but no matching "
                f"entry exists in core.grammar_meta.LANGUAGES"
            ) from e

    # ── required hooks ───────────────────────────────────────────────
    def extract_imports(
        self,
        file_version_id: int,
        source_rel_path: str,
        ts_root,
        db_id_for: dict[int, int],
    ) -> list[ImportEntry]:
        raise NotImplementedError(f"{type(self).__name__}.extract_imports")

    def resolve(
        self, entry: ImportEntry, idx: "BranchIndex", cfg: "LanguageConfig",
    ) -> ResolvedImport:
        raise NotImplementedError(f"{type(self).__name__}.resolve")

    # ── optional: branch-index contribution ──────────────────────────
    def init_state(self) -> Any:
        """Fresh state object stored at `idx.lang_state[self.name]`. Default
        None — languages without per-language indexes opt out."""
        return None

    def index_file(self, fvid: int, rel_path: str, state: Any) -> None:
        """Called once per file of this language during BranchIndex build."""
        return None

    def finalize_index(self, repo_root: Path | None, state: Any) -> None:
        """Called once after all files are indexed (e.g. read go.mod, walk Cargo.toml)."""
        return None

    # ── optional: cross-file linker hooks ────────────────────────────
    def seed_implicit_imports(self, idx: "BranchIndex") -> dict[int, set[int]]:
        """Return extra (importer_fvid → imported_fvids) edges intrinsic to
        this language's scoping rules — e.g. Go files in the same package
        see each other without explicit imports. Default: nothing."""
        return {}

    def expand_import_target(
        self, target_fvid: int, idx: "BranchIndex",
    ) -> set[int]:
        """Expand a single resolved import target into the set of files it
        effectively reaches. Used by the cross-file linker when one import
        targets a package containing multiple files (Go). Default: just the
        target itself."""
        return {target_fvid}

    # ── optional: semantic resolver hooks ────────────────────────────
    def should_skip_file(self, rel_path: str) -> bool:
        """Whole-file skip — return True to drop the file from semantic emission
        entirely. Rust uses this for `tests/` / `benches/` / `examples/` dirs."""
        return False

    def precompute_file_state(self, ts_walk: list, ctx: SemanticContext) -> None:
        """Stash per-file precomputed state into `ctx.scratch` before any other
        hook runs. Rust uses this to collect the set of ts_node ids under test
        gates (`#[cfg(test)]`, `#[test]`, `mod tests { … }`) — every emission
        loop in the resolver consults this set."""
        return None

    def resolve_reference(
        self, ts_node, rule, ctx: SemanticContext,
    ) -> "ResolvedReference | None":
        """Per-reference resolution override. Called before the YAML-driven
        inferred/certain path; return a `ResolvedReference` when AST
        context yields a more confident answer than name-only lookup, or
        None to fall through to the rule's declared strategy.
        """
        return None

    def qualified_name_prefix(self, ts_node, ctx: SemanticContext) -> str | None:
        """Return an extra qualified-name prefix segment for this node, or None.
        Rust uses this to prepend an impl block's target type to method names
        (`Counter::new` → qualified_name includes `Counter`)."""
        return None

    def synthesize_inheritance(
        self, ts_walk: list, ctx: SemanticContext,
    ) -> list[InheritanceEdge]:
        """Return language-specific inheritance edges the YAML-driven loop
        can't express. Rust uses this for `impl Trait for Type` and
        `#[derive(...)]` macro edges."""
        return []
