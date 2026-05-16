"""Deprecated shim: re-exports the JS/TS path-resolution helpers from their
new home under `core.languages._shared.node_resolution`. Removed in Phase 6
of the languages/ refactor — update imports to point at the new path.
"""

from __future__ import annotations

from .languages._shared.node_resolution import (  # noqa: F401
    NODE_EXTENSIONS,
    TsconfigPaths,
    is_relative_specifier,
    load_tsconfig_paths,
    match_path_alias,
    node_candidates,
    normalize_relative_posix,
    package_name_for_specifier,
    resolve_relative,
    resolve_tsconfig_alias,
)


__all__ = [
    "NODE_EXTENSIONS",
    "TsconfigPaths",
    "is_relative_specifier",
    "load_tsconfig_paths",
    "match_path_alias",
    "node_candidates",
    "normalize_relative_posix",
    "package_name_for_specifier",
    "resolve_relative",
    "resolve_tsconfig_alias",
]
