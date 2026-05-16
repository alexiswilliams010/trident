"""Repo file discovery, dependency-aware.

Yields source files for indexing while skipping dependency directories
(node_modules, lib, venv, ...). The set of dependency dirs is per-language
and is loaded from YAML configs in Phase 3. For Phase 1 we provide sensible
hard-coded defaults so the walker is usable before YAML configs exist.

User-defined exclusions are supported via two channels: a `.tridentignore`
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

`walk_repo_tree` is the Merkle-tree-aware variant used by the extractor: it
yields the same `DiscoveredFile` set and additionally returns a per-directory
SHA-256 over the sorted manifest of children. The leaf hashes feeding into
each directory hash come from a caller-supplied `leaf_hash_provider`, which
lets the extractor reuse cached content hashes when a file's size hasn't
changed.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

from .grammar_meta import LANGUAGES, language_for_path
from .languages import HANDLERS

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

# Default dependency dirs per language. Sourced from each LanguageHandler's
# `dependency_dirs` class attribute, so adding a language = one place to edit
# (the handler module under core/languages/<lang>/).
DEFAULT_DEP_PATHS: dict[str, tuple[str, ...]] = {
    name: handler.dependency_dirs
    for name, handler in HANDLERS.items()
    if handler.dependency_dirs
}

TRIDENT_IGNORE_FILE = ".tridentignore"

ROOT_DIR_PATH = ""  # branch_dirs stores the repo root with empty-string path.

# Leaf provider contract: given (rel_path, size, abs_path), return the
# file's SHA-256 hex digest. The extractor implements this with a size-cache
# shortcut; tests and the basic `walk_repo` path bypass tree-building and
# don't call any provider.
LeafHashProvider = Callable[[str, int, Path], str]


@dataclass
class WalkConfig:
    repo_root: Path
    # Language → set of relative directory names to prune during walk.
    dep_paths: dict[str, set[str]] = field(default_factory=dict)
    # If True, also skip dotfile dirs (e.g. .next, .cache).
    skip_hidden: bool = True
    # User-defined exclusion patterns (from .tridentignore + --exclude flags).
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


def read_tridentignore(repo_root: str | Path) -> tuple[str, ...]:
    """Return the list of patterns from `.tridentignore` at the repo root.

    Empty lines and lines starting with `#` are skipped. Trailing slashes are
    stripped (the directory-vs-file distinction is handled by the caller).
    Returns an empty tuple if the file is absent.
    """
    path = Path(repo_root) / TRIDENT_IGNORE_FILE
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
    size: int            # bytes (from os.stat); cached on branch_files for the size-cache shortcut


def _scan_one_dir(
    config: WalkConfig,
    abs_dir: Path,
    rel_dir: str,
) -> tuple[list[DiscoveredFile], list[tuple[Path, str, str]]]:
    """List the immediate children of `abs_dir` that survive the filters.

    Returns (files, subdirs). `subdirs` items are (abs_path, rel_path, name).
    Filenames in `files` are sorted by basename; subdirs are sorted by name.
    Sorting matters for the Merkle tree's canonical dir manifest.
    """
    skip_dirs = ALWAYS_IGNORED | config.all_dep_dirs()
    excludes = config.exclude_patterns

    files: list[DiscoveredFile] = []
    subdirs: list[tuple[Path, str, str]] = []

    with os.scandir(abs_dir) as it:
        entries = list(it)
    entries.sort(key=lambda e: e.name)

    for entry in entries:
        name = entry.name
        rel = f"{rel_dir}/{name}" if rel_dir else name
        if entry.is_dir(follow_symlinks=False):
            if name in skip_dirs:
                continue
            if config.skip_hidden and name.startswith("."):
                continue
            if _matches_excludes(rel, excludes):
                continue
            subdirs.append((Path(entry.path), rel, name))
        elif entry.is_file(follow_symlinks=False):
            lang = language_for_path(Path(entry.path))
            if lang is None:
                continue
            if _matches_excludes(rel, excludes):
                continue
            try:
                size = entry.stat(follow_symlinks=False).st_size
            except FileNotFoundError:
                continue
            files.append(DiscoveredFile(
                path=Path(entry.path), rel_path=rel, language=lang, size=size,
            ))

    return files, subdirs


def walk_repo(config: WalkConfig) -> Iterator[DiscoveredFile]:
    """Yield source files in `config.repo_root`, pruning dep + ignored dirs
    and applying any user `exclude_patterns`. Does not compute tree hashes."""
    root = config.repo_root
    if not root.is_dir():
        raise ValueError(f"repo_root not a directory: {root}")

    stack: list[tuple[Path, str]] = [(root, "")]
    while stack:
        abs_dir, rel_dir = stack.pop()
        files, subdirs = _scan_one_dir(config, abs_dir, rel_dir)
        for f in files:
            yield f
        # Push in reverse so leftmost dir is processed next.
        for abs_sub, rel_sub, _name in reversed(subdirs):
            stack.append((abs_sub, rel_sub))


def _dir_manifest_hash(entries: list[tuple[str, str, str]]) -> str:
    """SHA-256 over the canonical manifest of one directory.

    `entries` is a list of (kind, name, hash) where kind is 'f' or 'd'.
    Caller passes them already sorted by name. Lines are joined with `\n`,
    fields with `|`. Names are NUL-rejected to keep the format unambiguous;
    POSIX filenames cannot contain NUL anyway.
    """
    h = hashlib.sha256()
    for kind, name, child_hash in entries:
        if "\x00" in name:
            raise ValueError(f"filename contains NUL: {name!r}")
        h.update(f"{kind}|{name}|{child_hash}\n".encode())
    return h.hexdigest()


def walk_repo_tree(
    config: WalkConfig,
    leaf_hash_provider: LeafHashProvider,
) -> tuple[list[DiscoveredFile], dict[str, str]]:
    """Walk the repo and build the per-directory Merkle hashes.

    Returns (files, dir_hashes). `files` is the same set `walk_repo` would
    yield, in deterministic (depth-first, name-sorted) order. `dir_hashes`
    maps every visited directory's repo-relative path (with the root as
    `ROOT_DIR_PATH`/empty string) to its SHA-256 manifest hash.

    `leaf_hash_provider(rel_path, size, abs_path)` is called for every file
    and must return the file's content hash (hex SHA-256). The extractor
    uses this hook to reuse cached hashes when a file's size hasn't changed.
    """
    root = config.repo_root
    if not root.is_dir():
        raise ValueError(f"repo_root not a directory: {root}")

    files: list[DiscoveredFile] = []
    dir_hashes: dict[str, str] = {}

    def recurse(abs_dir: Path, rel_dir: str) -> str:
        local_files, subdirs = _scan_one_dir(config, abs_dir, rel_dir)
        # Entries fed into this dir's hash, sorted by name. Files and subdirs
        # were each independently sorted in _scan_one_dir; merge by name.
        manifest: list[tuple[str, str, str]] = []
        # Build child hashes first so they're available for the manifest.
        sub_hashes: dict[str, str] = {}
        for abs_sub, rel_sub, name in subdirs:
            sub_hashes[name] = recurse(abs_sub, rel_sub)
        file_hashes: dict[str, str] = {}
        for f in local_files:
            file_hashes[f.path.name] = leaf_hash_provider(f.rel_path, f.size, f.path)
            files.append(f)
        for name in sorted(set(sub_hashes) | set(file_hashes)):
            if name in file_hashes:
                manifest.append(("f", name, file_hashes[name]))
            else:
                manifest.append(("d", name, sub_hashes[name]))
        h = _dir_manifest_hash(manifest)
        dir_hashes[rel_dir] = h
        return h

    recurse(root, ROOT_DIR_PATH)
    return files, dir_hashes


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
        try:
            size = full.stat().st_size
        except FileNotFoundError:
            continue
        yield DiscoveredFile(
            path=full,
            rel_path=full.relative_to(root).as_posix(),
            language=lang,
            size=size,
        )


__all__ = [
    "ALWAYS_IGNORED",
    "DEFAULT_DEP_PATHS",
    "DiscoveredFile",
    "LeafHashProvider",
    "ROOT_DIR_PATH",
    "TRIDENT_IGNORE_FILE",
    "WalkConfig",
    "read_tridentignore",
    "walk_dependency_files",
    "walk_repo",
    "walk_repo_tree",
    "LANGUAGES",
]
