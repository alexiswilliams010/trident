"""Repo file discovery, dependency-aware.

Yields source files for indexing while skipping dependency directories
(node_modules, lib, venv, ...). The set of dependency dirs is per-language
and is loaded from YAML configs in Phase 3. For Phase 1 we provide sensible
hard-coded defaults so the walker is usable before YAML configs exist.

User-defined exclusions are supported via two channels: a `.tsgrepignore`
file at the repo root and per-invocation `--exclude` CLI flags. Patterns are
fnmatch-style globs:

  - Patterns containing `/` match against the full repo-relative path
    (e.g. `src/test/*` matches files under that exact directory).
  - Patterns without `/` match against any path component (e.g. `test`
    matches every directory or file named `test` at any depth; `*.t.sol`
    matches every Foundry test file anywhere).

A second method, `walk_dependency_files`, is used by Phase 3's targeted
dependency pass to read explicitly-named files inside dep dirs (e.g. an
import resolver tells us "we need lib/forge-std/src/Test.sol" — that file
gets yielded even though `lib/` is otherwise pruned). Exclusion patterns
do not apply to that pass.
"""

from __future__ import annotations

import fnmatch
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
    "go": ("vendor",),
}

TSGREP_IGNORE_FILE = ".tsgrepignore"


@dataclass
class WalkConfig:
    repo_root: Path
    # Language → set of relative directory names to prune during walk.
    dep_paths: dict[str, set[str]] = field(default_factory=dict)
    # If True, also skip dotfile dirs (e.g. .next, .cache).
    skip_hidden: bool = True
    # User-defined exclusion patterns (from .tsgrepignore + --exclude flags).
    exclude_patterns: tuple[str, ...] = ()

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


def read_tsgrepignore(repo_root: str | Path) -> tuple[str, ...]:
    """Return the list of patterns from `.tsgrepignore` at the repo root.

    Empty lines and lines starting with `#` are skipped. Trailing slashes are
    stripped (the directory-vs-file distinction is handled by the caller).
    Returns an empty tuple if the file is absent.
    """
    path = Path(repo_root) / TSGREP_IGNORE_FILE
    if not path.is_file():
        return ()
    out: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("/"):
            line = line[:-1]
        out.append(line)
    return tuple(out)


def _matches_excludes(rel_path: str, patterns: tuple[str, ...]) -> bool:
    """True if any pattern matches `rel_path` (forward-slash separated).

    Convention:
      - Pattern with `/` → match against full rel_path (fnmatch).
      - Pattern without `/` → match against any path component (basename match).
    """
    if not patterns:
        return False
    components = rel_path.split("/")
    for pat in patterns:
        if "/" in pat:
            if fnmatch.fnmatchcase(rel_path, pat):
                return True
        else:
            if any(fnmatch.fnmatchcase(c, pat) for c in components):
                return True
    return False


@dataclass
class DiscoveredFile:
    path: Path           # absolute path
    rel_path: str        # path relative to repo_root, forward-slash separated
    language: str


def walk_repo(config: WalkConfig) -> Iterator[DiscoveredFile]:
    """Yield source files in `config.repo_root`, pruning dep + ignored dirs
    and applying any user `exclude_patterns`."""
    root = config.repo_root
    if not root.is_dir():
        raise ValueError(f"repo_root not a directory: {root}")

    skip_dirs = ALWAYS_IGNORED | config.all_dep_dirs()
    excludes = config.exclude_patterns

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        if rel_dir == ".":
            rel_dir = ""

        # Prune in-place so os.walk does not descend.
        pruned: list[str] = []
        for d in list(dirnames):
            if d in skip_dirs:
                continue
            if config.skip_hidden and d.startswith("."):
                continue
            sub_rel = f"{rel_dir}/{d}" if rel_dir else d
            if _matches_excludes(sub_rel, excludes):
                continue
            pruned.append(d)
        dirnames[:] = pruned

        for fname in filenames:
            full = Path(dirpath) / fname
            lang = language_for_path(full)
            if lang is None:
                continue
            rel = full.relative_to(root).as_posix()
            if _matches_excludes(rel, excludes):
                continue
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
    "TSGREP_IGNORE_FILE",
    "WalkConfig",
    "read_tsgrepignore",
    "walk_dependency_files",
    "walk_repo",
    "LANGUAGES",
]
