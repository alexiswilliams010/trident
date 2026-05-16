"""Import resolution for Go."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import ImportEntry, ResolvedImport

if TYPE_CHECKING:
    from ...heuristic_resolver import BranchIndex


def resolve_go(entry: ImportEntry, idx: "BranchIndex") -> ResolvedImport:
    """Classify a Go import path.

    Three buckets, in priority order:
      - intra_repo: matches the module path declared in go.mod, suffix maps
        to a known package directory.
      - external (stdlib): first segment has no dot ("fmt", "net/http", …).
      - external (third-party): everything else, e.g. github.com/x/y. The
        package_name keeps the org/repo prefix so the same dependency rolls
        up across multiple subpackage imports.
    """
    raw = entry.import_path
    go_state = idx.lang_state.get("go")
    mod = go_state.module_path if go_state is not None else None
    if mod and (raw == mod or raw.startswith(mod + "/")):
        suffix = "" if raw == mod else raw[len(mod) + 1 :]
        target = go_state.pkg_index.get(suffix) if go_state is not None else None
        if target is not None:
            return ResolvedImport(entry, "intra_repo", resolved_file_version_id=target)
        return ResolvedImport(entry, "unresolved")

    first = raw.split("/", 1)[0]
    if "." not in first:
        return ResolvedImport(entry, "external", package_name=raw)

    parts = raw.split("/")
    if first in {"github.com", "gitlab.com", "bitbucket.org"} and len(parts) >= 3:
        pkg = "/".join(parts[:3])
    else:
        pkg = parts[0]
    return ResolvedImport(entry, "external", package_name=pkg)
