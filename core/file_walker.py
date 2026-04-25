"""Repo file discovery, dependency-aware.

Yields source files for indexing while skipping dependency directories
(node_modules, lib, venv, ...). The set of dependency dirs is per-language
and is loaded from YAML configs in Phase 3. For Phase 1 we provide sensible
hard-coded defaults so the walker is usable before YAML configs exist.

A second method, `walk_dependency_files`, is used by Phase 3's targeted
dependency pass to read explicitly-named files inside dep dirs (e.g. an
import resolver tells us "we need lib/forge-std/src/Test.sol" — that file
gets yielded even though `lib/` is otherwise pruned).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from .grammar_meta import LANGUAGES, language_for_path

# Directories pruned everywhere, regardless of language.
ALWAYS_IGNORED: frozenset[str] = frozenset({
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
})

# Default dependency dirs per language. Overridable by passing dep_paths
# explicitly (Phase 3 will load these from YAML configs).
DEFAULT_DEP_PATHS: dict[str, tuple[str, ...]] = {
    "python": ("venv", ".venv", "site-packages", "env", ".env"),
    "solidity": ("lib", "node_modules", "out", "cache", "artifacts"),
}


@dataclass
class WalkConfig:
    repo_root: Path
    # Language → set of relative directory names to prune during walk.
    dep_paths: dict[str, set[str]] = field(default_factory=dict)
    # If True, also skip dotfile dirs (e.g. .next, .cache).
    skip_hidden: bool = True

    @classmethod
    def with_defaults(cls, repo_root: str | Path) -> "WalkConfig":
        return cls(
            repo_root=Path(repo_root).resolve(),
            dep_paths={lang: set(paths) for lang, paths in DEFAULT_DEP_PATHS.items()},
        )

    def all_dep_dirs(self) -> set[str]:
        merged: set[str] = set()
        for paths in self.dep_paths.values():
            merged |= paths
        return merged


@dataclass
class DiscoveredFile:
    path: Path           # absolute path
    rel_path: str        # path relative to repo_root, forward-slash separated
    language: str


def walk_repo(config: WalkConfig) -> Iterator[DiscoveredFile]:
    """Yield source files in `config.repo_root`, pruning dep + ignored dirs."""
    root = config.repo_root
    if not root.is_dir():
        raise ValueError(f"repo_root not a directory: {root}")

    skip_dirs = ALWAYS_IGNORED | config.all_dep_dirs()

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune in-place so os.walk does not descend.
        pruned: list[str] = []
        for d in list(dirnames):
            if d in skip_dirs:
                continue
            if config.skip_hidden and d.startswith("."):
                continue
            pruned.append(d)
        dirnames[:] = pruned

        for fname in filenames:
            full = Path(dirpath) / fname
            lang = language_for_path(full)
            if lang is None:
                continue
            rel = full.relative_to(root).as_posix()
            yield DiscoveredFile(path=full, rel_path=rel, language=lang)


def walk_dependency_files(
    repo_root: str | Path,
    candidate_paths: Iterable[str],
) -> Iterator[DiscoveredFile]:
    """Yield specific files inside dependency dirs (Phase 3 targeted pass).

    `candidate_paths` are relative-to-repo paths produced by the heuristic
    resolver when an external import resolves to a concrete file, e.g.
    "lib/forge-std/src/Test.sol". Only those files are returned — sibling
    files in the same dep package stay unparsed.
    """
    root = Path(repo_root).resolve()
    seen: set[Path] = set()
    for rel in candidate_paths:
        full = (root / rel).resolve()
        if full in seen:
            continue
        seen.add(full)
        if not full.is_file():
            continue
        # Confine to repo_root (no escaping via "../..").
        try:
            full.relative_to(root)
        except ValueError:
            continue
        lang = language_for_path(full)
        if lang is None:
            continue
        yield DiscoveredFile(
            path=full,
            rel_path=full.relative_to(root).as_posix(),
            language=lang,
        )


__all__ = [
    "ALWAYS_IGNORED",
    "DEFAULT_DEP_PATHS",
    "DiscoveredFile",
    "WalkConfig",
    "walk_dependency_files",
    "walk_repo",
    "LANGUAGES",
]
