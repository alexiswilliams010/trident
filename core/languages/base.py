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
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from ..config_loader import LanguageConfig
    from ..grammar_meta import LanguageSpec
    from ..heuristic_resolver import BranchIndex, ImportEntry, ResolvedImport


@dataclass
class SemanticContext:
    """Per-file context passed to semantic-resolver hooks.

    `scratch` is a free-form slot handlers can write into during
    `precompute_file_state` and read back from later hooks within the same
    file (e.g. Rust stashes test-skip node IDs here).
    """

    config: "LanguageConfig"
    file_path: str
    scratch: dict[str, Any] = field(default_factory=dict)


@dataclass
class InheritanceEdge:
    """Synthesized inheritance relation: subclass extends/implements `base_qualified_name`."""

    subclass_def_id: int
    base_qualified_name: str
    confidence: str = "certain"


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
        self, tree_root, file_version_id: int, rel_path: str,
    ) -> Iterable["ImportEntry"]:
        raise NotImplementedError(f"{type(self).__name__}.extract_imports")

    def resolve(
        self, entry: "ImportEntry", idx: "BranchIndex", cfg: "LanguageConfig",
    ) -> "ResolvedImport":
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

    # ── optional: semantic resolver hooks ────────────────────────────
    def should_skip_file(self, rel_path: str) -> bool:
        return False

    def precompute_file_state(self, tree_root, ctx: SemanticContext) -> None:
        return None

    def qualified_name_prefix(self, ts_node, ctx: SemanticContext) -> str | None:
        return None

    def synthesize_inheritance(
        self, ts_node, ctx: SemanticContext,
    ) -> list[InheritanceEdge]:
        return []
